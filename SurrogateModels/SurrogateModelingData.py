from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from SurrogateModelingConfig import FlowCaseConfig
from SurrogateModelingUtils import (
    _pick,
    d1_periodic_with_overlap,
    hard_project_theta_periodic,
    neighbor_minus,
    neighbor_plus,
)


def make_pure_physics_debug_case(
    *,
    blade_params: str | Path,
    n: int = 64,
    rh: float | None = None,
    rs: float | None = None,
    h: float | None = None,
    mu: float = 0.006,
    rho: float = 10650.0,
    omega: float = -420.0 * np.pi / 60.0,
    qv: float = 0.16,
    n_blade: int | None = None,
    z0: float | None = None,
    g: float = 9.8,
) -> dict[str, Any]:
    # 纯物理调试模式下，数据标签可以没有，但几何和工况必须齐全。
    blade_params = Path(blade_params)
    with blade_params.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    global_params = params.get("global_parameters", {})
    return {
        "n": int(n),
        "rh": float(rh if rh is not None else global_params.get("hub_radius")),
        "rs": float(rs if rs is not None else global_params.get("shroud_radius")),
        "h": float(h if h is not None else global_params.get("passage_height_H1")),
        "mu": float(mu),
        "rho": float(rho),
        "omega": float(omega),
        "qv": float(qv),
        "n_blade": int(n_blade if n_blade is not None else global_params.get("blade_count_N")),
        "z0": float(z0 if z0 is not None else global_params.get("z0", 0.0)),
        "g": float(g),
        "absolute_frame": True,
        "blade_params": str(blade_params),
    }


def _normalize_blade_params_inputs(
    blade_params: str | Path | Sequence[str | Path],
) -> list[Path]:
    # 训练时允许同时输入多个叶片几何文件。
    # 这里统一展开成 Path 列表，后面的建 case 流程就可以完全复用。
    if isinstance(blade_params, (str, Path)):
        items = [Path(blade_params)]
    else:
        items = [Path(item) for item in blade_params]
    if len(items) == 0:
        raise ValueError("blade_params must contain at least one blade geometry file.")
    return items


def make_pure_physics_debug_cases(
    *,
    blade_params: str | Path | Sequence[str | Path],
    n: int = 64,
    rh: float | None = None,
    rs: float | None = None,
    h: float | None = None,
    mu: float = 0.006,
    rho: float = 10650.0,
    omega: float = -420.0 * np.pi / 60.0,
    qv: float = 0.16,
    n_blade: int | None = None,
    z0: float | None = None,
    g: float = 9.8,
) -> list[dict[str, Any]]:
    # 这是“同一工况、多叶型训练”的直接入口。
    # 除了 blade_params 可以是一组之外，其他工况和几何尺度参数都保持一致。
    cases: list[dict[str, Any]] = []
    for blade_path in _normalize_blade_params_inputs(blade_params):
        cases.append(
            make_pure_physics_debug_case(
                blade_params=blade_path,
                n=n,
                rh=rh,
                rs=rs,
                h=h,
                mu=mu,
                rho=rho,
                omega=omega,
                qv=qv,
                n_blade=n_blade,
                z0=z0,
                g=g,
            )
        )
    return cases


def find_unique_simulation_csv(folder: str | Path) -> Path:
    folder = Path(folder)
    matches = sorted(folder.glob("*.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CSV file in {folder}, found {len(matches)}.")
    return matches[0]


def _load_blade_global_parameters(blade_params: str | Path) -> Mapping[str, Any]:
    with Path(blade_params).open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    return params.get("global_parameters", {})


def _canonical_csv_columns(columns: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in columns:
        key = str(column).strip().lower().replace("_", "-")
        key = " ".join(key.split())
        result[key] = str(column)
    return result


def _csv_column(columns: Mapping[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", "-")
        key = " ".join(key.split())
        if key in columns:
            return columns[key]
    return None


def _infer_cylindrical_frame_from_xyz(
    df,
    *,
    rh: float,
    rs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], str, dict[str, Any]]:
    columns = _canonical_csv_columns(df.columns)
    x_col = _csv_column(columns, "x-coordinate", "x")
    y_col = _csv_column(columns, "y-coordinate", "y")
    if x_col is None or y_col is None:
        raise ValueError("CSV must provide x/y coordinates for cylindrical projection.")

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    origin_radius = np.sqrt(x ** 2 + y ** 2)
    origin_q = np.quantile(origin_radius, [0.01, 0.99])
    radial_span = max(rs - rh, 1e-12)
    origin_matches = (
        abs(origin_q[0] - rh) < 0.25 * radial_span
        and abs(origin_q[1] - rs) < 0.25 * radial_span
    )

    if origin_matches:
        cx = 0.0
        cy = 0.0
        source = "origin-z-axis"
    else:
        # Fluent/CAD exports may carry the annular passage in a translated
        # x-y frame. We recenter before cylindrical projection so the local
        # cylindrical axis is exactly the z axis used by the surrogate grid.
        cx = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
        cy = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
        source = "shifted-to-z-axis"

    x_local = x - cx
    y_local = y - cy
    radius = np.sqrt(x_local ** 2 + y_local ** 2)
    theta = np.mod(np.arctan2(y_local, x_local), 2.0 * np.pi)
    local_q = np.quantile(radius, [0.01, 0.99])
    diagnostics = {
        "origin_radius_q01": float(origin_q[0]),
        "origin_radius_q99": float(origin_q[1]),
        "local_radius_q01": float(local_q[0]),
        "local_radius_q99": float(local_q[1]),
        "axis_offset": float(np.sqrt(cx ** 2 + cy ** 2)),
        "origin_matches_annulus": bool(origin_matches),
    }
    return radius, theta, x_local, y_local, (float(cx), float(cy)), source, diagnostics


def _cartesian_velocity_from_csv(
    df,
    columns: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    ux_col = _csv_column(columns, "x-velocity", "velocity-x", "x velocity", "u-x", "ux")
    uy_col = _csv_column(columns, "y-velocity", "velocity-y", "y velocity", "u-y", "uy")
    uz_col = _csv_column(columns, "z-velocity", "velocity-z", "z velocity", "u-z", "uz")
    if ux_col is not None and uy_col is not None and uz_col is not None:
        return (
            df[ux_col].to_numpy(dtype=float),
            df[uy_col].to_numpy(dtype=float),
            df[uz_col].to_numpy(dtype=float),
            "xyz-columns",
        )

    # The current Fluent export names the three Cartesian components with
    # axial/radial/tangential labels. In cartesian mode we interpret them as
    # x/y/z components as requested by the main training workflow.
    ux_col = _csv_column(columns, "radial-velocity")
    uy_col = _csv_column(columns, "tangential-velocity")
    uz_col = _csv_column(columns, "axial-velocity")
    if ux_col is not None and uy_col is not None and uz_col is not None:
        return (
            df[ux_col].to_numpy(dtype=float),
            df[uy_col].to_numpy(dtype=float),
            df[uz_col].to_numpy(dtype=float),
            "legacy-radial-tangential-axial-as-xyz",
        )

    raise ValueError("CSV must provide x/y/z velocity columns for cartesian velocity conversion.")


def _cylindrical_velocity_from_cartesian(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    radius: np.ndarray,
    x_local: np.ndarray,
    y_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_radius = np.maximum(radius, 1e-12)
    cos_theta = x_local / safe_radius
    sin_theta = y_local / safe_radius
    ur = ux * cos_theta + uy * sin_theta
    ut = -ux * sin_theta + uy * cos_theta
    return ur, ut, uz


def _interpolate_scattered_fields_to_targets(
    points: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    target_points: np.ndarray,
    output_shape: Sequence[int],
    chunk_size: int = 250_000,
) -> dict[str, np.ndarray]:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    from scipy.spatial import Delaunay

    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        raise ValueError("Not enough CSV points inside the selected blade passage sector.")
    target_points = np.asarray(target_points, dtype=np.float64)

    output_shape = tuple(int(item) for item in output_shape)
    if int(np.prod(output_shape)) != target_points.shape[0]:
        raise ValueError("output_shape must match the number of interpolation target points.")

    tri = Delaunay(points)
    result: dict[str, np.ndarray] = {}
    for name, field_values in values.items():
        field_values = np.asarray(field_values, dtype=np.float64)
        linear = LinearNDInterpolator(tri, field_values, fill_value=np.nan)
        nearest = NearestNDInterpolator(points, field_values)
        out = np.empty(target_points.shape[0], dtype=np.float32)

        for start in range(0, target_points.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), target_points.shape[0])
            chunk = target_points[start:end]
            interpolated = np.asarray(linear(chunk), dtype=np.float64)
            missing = ~np.isfinite(interpolated)
            if np.any(missing):
                interpolated[missing] = nearest(chunk[missing])
            out[start:end] = interpolated.astype(np.float32)

        tensor_field = out.reshape(output_shape)
        theta_dim = 1 if len(output_shape) == 3 else 0
        tensor_field = hard_project_theta_periodic(torch.from_numpy(tensor_field), theta_dim=theta_dim).numpy()
        result[name] = tensor_field.astype(np.float32, copy=False)
    return result


def _interpolate_scattered_fields_to_grid(
    points: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    n: int,
    chunk_size: int = 250_000,
) -> dict[str, np.ndarray]:
    target_axis = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    rr, tt, zz = np.meshgrid(target_axis, target_axis, target_axis, indexing="ij")
    target_points = np.column_stack([rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)])
    return _interpolate_scattered_fields_to_targets(
        points,
        values,
        target_points=target_points,
        output_shape=(int(n), int(n), int(n)),
        chunk_size=chunk_size,
    )


def _interpolate_scattered_fields_to_span(
    points: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    span: float,
    n: int,
    chunk_size: int = 250_000,
) -> dict[str, np.ndarray]:
    target_axis = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    tt, zz = np.meshgrid(target_axis, target_axis, indexing="ij")
    rr = np.full_like(tt, float(np.clip(span, 0.0, 1.0)))
    target_points = np.column_stack([rr.reshape(-1), tt.reshape(-1), zz.reshape(-1)])
    return _interpolate_scattered_fields_to_targets(
        points,
        values,
        target_points=target_points,
        output_shape=(int(n), int(n)),
        chunk_size=chunk_size,
    )


def _span_slice_from_grid(field: Any, span: float) -> np.ndarray:
    # span 是无量纲半径位置：0=hub，1=shroud。若不是网格点，按 R 方向线性插值。
    array = field.detach().cpu().numpy() if torch.is_tensor(field) else np.asarray(field)
    if array.ndim != 3:
        raise ValueError("span slicing expects a 3D field with shape [R, Theta, Z].")
    n = array.shape[0]
    position = float(np.clip(span, 0.0, 1.0)) * max(n - 1, 0)
    lower = int(np.floor(position))
    upper = min(lower + 1, n - 1)
    weight = position - lower
    if upper == lower:
        return array[lower].astype(np.float32, copy=False)
    return ((1.0 - weight) * array[lower] + weight * array[upper]).astype(np.float32, copy=False)


def _physical_pressure_gradient_magnitude(
    pressure: Any,
    config: FlowCaseConfig,
) -> np.ndarray:
    # CFD 导出的压力绝对值可能带任意参考零点；用于评价时只看物理空间中的柱坐标梯度幅值。
    # 输出单位为 Pa/m，对应 sqrt((dP/dr)^2 + (1/r dP/dtheta)^2 + (dP/dz)^2)。
    p = torch.as_tensor(pressure, dtype=torch.float32).unsqueeze(0)
    dR = torch.tensor(config.dR, dtype=torch.float32).view(1, 1, 1, 1)
    dTheta = torch.tensor(config.dTheta, dtype=torch.float32).view(1, 1, 1, 1)
    dZ = torch.tensor(config.dZ, dtype=torch.float32).view(1, 1, 1, 1)

    dR_p = (neighbor_plus(p, dim=1, periodic=False) - neighbor_minus(p, dim=1, periodic=False)) / (2.0 * dR)
    dTheta_p = d1_periodic_with_overlap(p, dim=2, spacing=dTheta)
    dZ_p = (neighbor_plus(p, dim=3, periodic=False) - neighbor_minus(p, dim=3, periodic=False)) / (2.0 * dZ)

    r_norm = torch.linspace(0.0, 1.0, p.shape[1], dtype=torch.float32).view(1, -1, 1, 1)
    r_phys = config.rh + r_norm * config.delta_r
    dp_dr = dR_p / max(config.delta_r, 1e-12)
    dp_dtheta_over_r = dTheta_p / torch.clamp(r_phys * config.theta0, min=1e-12)
    dp_dz = dZ_p / max(config.h, 1e-12)
    grad_mag = torch.sqrt(dp_dr ** 2 + dp_dtheta_over_r ** 2 + dp_dz ** 2)
    return grad_mag[0].detach().cpu().numpy().astype(np.float32, copy=False)


def make_supervised_simulation_case(
    folder: str | Path,
    *,
    n: int = 64,
    mu: float = 0.006,
    rho: float = 10650.0,
    omega: float = -210.0 * 2.0 * np.pi / 60.0,
    qv: float = 0.025,
    g: float = 9.8,
    theta_sector_index: int = 0,
    csv_path: str | Path | None = None,
    interpolation_chunk_size: int = 250_000,
    verbose_geometry_check: bool = True,
) -> dict[str, Any]:
    import pandas as pd

    folder = Path(folder)
    blade_params = folder / "blade_params.json"
    if not blade_params.exists():
        raise FileNotFoundError(f"Missing blade_params.json in {folder}.")
    csv_file = Path(csv_path) if csv_path is not None else find_unique_simulation_csv(folder)

    global_params = _load_blade_global_parameters(blade_params)
    rh = float(global_params["hub_radius"])
    rs = float(global_params["shroud_radius"])
    h = float(global_params["passage_height_H1"])
    n_blade = int(global_params["blade_count_N"])
    z0 = float(global_params.get("z0", 0.0))
    theta0 = 2.0 * np.pi / n_blade

    df = pd.read_csv(csv_file)
    df.columns = [str(column).strip() for column in df.columns]
    columns = _canonical_csv_columns(df.columns)
    pressure_col = _csv_column(columns, "absolute-pressure", "pressure", "static-pressure", "p")
    z_col = _csv_column(columns, "axial-coordinate", "z-coordinate", "z")
    if any(item is None for item in [pressure_col, z_col]):
        raise ValueError(f"CSV {csv_file} is missing one or more required flow-field columns.")

    radius, theta, x_local, y_local, xy_center, radius_source, geometry_diagnostics = _infer_cylindrical_frame_from_xyz(
        df,
        rh=rh,
        rs=rs,
    )
    ux, uy, uz_xyz, velocity_source = _cartesian_velocity_from_csv(df, columns)
    ur_values, ut_values, uz_values = _cylindrical_velocity_from_cartesian(
        ux,
        uy,
        uz_xyz,
        radius,
        x_local,
        y_local,
    )
    if verbose_geometry_check:
        print(
            "CSV cylinder check: local cylindrical frame "
            f"r[q01,q99]=({geometry_diagnostics['local_radius_q01']:.6g}, "
            f"{geometry_diagnostics['local_radius_q99']:.6g}), "
            f"design=({rh:.6g}, {rs:.6g}), "
            f"axis_xy=({xy_center[0]:.6g}, {xy_center[1]:.6g}), "
            f"frame={radius_source}, velocity={velocity_source}. "
            "q01/q99 only describe sampled coverage, not exact wall radii."
        )
    z_physical = df[z_col].to_numpy(dtype=float)
    radial_tol = 0.02 * max(rs - rh, 1e-12)
    sector_start = int(theta_sector_index) * theta0
    sector_theta = np.mod(theta - sector_start, 2.0 * np.pi)
    sector_mask = (
        (radius >= rh - radial_tol)
        & (radius <= rs + radial_tol)
        & (sector_theta >= 0.0)
        & (sector_theta <= theta0)
    )

    if int(np.sum(sector_mask)) < 128:
        raise ValueError(
            f"Only {int(np.sum(sector_mask))} CSV points found in sector {theta_sector_index}; "
            "check the coordinate origin or theta_sector_index."
        )

    z_sector = z_physical[sector_mask]
    z_min = float(np.nanmin(z_sector))
    z_max = float(np.nanmax(z_sector))
    z_span = max(z_max - z_min, 1e-12)
    if abs(z_span - h) > 0.15 * max(h, 1e-12):
        print(f"Warning: CSV axial span {z_span:.6g} differs from blade passage height {h:.6g}.")

    points = np.column_stack(
        [
            np.clip((radius[sector_mask] - rh) / max(rs - rh, 1e-12), 0.0, 1.0),
            np.clip(sector_theta[sector_mask] / theta0, 0.0, 1.0),
            np.clip((z_sector - z_min) / z_span, 0.0, 1.0),
        ]
    )
    field_values = {
        "UR": ur_values[sector_mask],
        "UT": ut_values[sector_mask],
        "UZ": uz_values[sector_mask],
        "P": df[pressure_col].to_numpy(dtype=float)[sector_mask],
    }
    interpolated = _interpolate_scattered_fields_to_grid(
        points,
        field_values,
        n=int(n),
        chunk_size=interpolation_chunk_size,
    )

    return {
        "n": int(n),
        "rh": rh,
        "rs": rs,
        "h": h,
        "mu": float(mu),
        "rho": float(rho),
        "omega": float(omega),
        "qv": float(qv),
        "n_blade": n_blade,
        "z0": z_min,
        "g": float(g),
        "absolute_frame": True,
        "blade_params": str(blade_params),
        "fields_are_dimensionless": False,
        "UR": interpolated["UR"],
        "UT": interpolated["UT"],
        "UZ": interpolated["UZ"],
        "P": interpolated["P"],
        "simulation_csv": str(csv_file),
        "csv_sector_index": int(theta_sector_index),
        "csv_points_in_sector": int(np.sum(sector_mask)),
        "csv_xy_center": tuple(float(v) for v in xy_center),
        "csv_radius_source": radius_source,
        "csv_velocity_source": velocity_source,
        "csv_geometry_diagnostics": geometry_diagnostics,
        "csv_blade_z0": z0,
        "csv_z_min": z_min,
        "csv_z_max": z_max,
    }


def make_supervised_simulation_cases(
    folders: str | Path | Sequence[str | Path],
    *,
    n: int = 64,
    mu: float = 0.006,
    rho: float = 10650.0,
    omega: float = -210.0 * 2.0 * np.pi / 60.0,
    qv: float = 0.025,
    g: float = 9.8,
    theta_sector_index: int = 0,
    interpolation_chunk_size: int = 250_000,
) -> list[dict[str, Any]]:
    if isinstance(folders, (str, Path)):
        folder_list = [Path(folders)]
    else:
        folder_list = [Path(folder) for folder in folders]
    if not folder_list:
        raise ValueError("folders must contain at least one simulation case folder.")

    return [
        make_supervised_simulation_case(
            folder,
            n=n,
            mu=mu,
            rho=rho,
            omega=omega,
            qv=qv,
            g=g,
            theta_sector_index=theta_sector_index,
            interpolation_chunk_size=interpolation_chunk_size,
        )
        for folder in folder_list
    ]


def _load_cfd_passage_scatter_from_case(
    case: Mapping[str, Any],
    *,
    theta_sector_index: int | None = None,
    verbose_geometry_check: bool = False,
) -> dict[str, Any]:
    # 后处理阶段需要从原始 CSV 直接插值到某个 span 平面。
    # 这里刻意复用训练导入时的坐标中心、单流道裁剪和 xyz->柱坐标速度转换规则。
    import pandas as pd

    csv_file_raw = _pick(case, "simulation_csv")
    blade_params_raw = _pick(case, "blade_params")
    if csv_file_raw is None or blade_params_raw is None:
        raise ValueError("CFD span plotting requires case['simulation_csv'] and case['blade_params'].")

    csv_file = Path(csv_file_raw)
    blade_params = Path(blade_params_raw)
    config = FlowCaseConfig.from_mapping(case)
    theta0 = config.theta0
    sector_index = int(theta_sector_index if theta_sector_index is not None else _pick(case, "csv_sector_index", default=0))

    df = pd.read_csv(csv_file)
    df.columns = [str(column).strip() for column in df.columns]
    columns = _canonical_csv_columns(df.columns)
    pressure_col = _csv_column(columns, "absolute-pressure", "pressure", "static-pressure", "p")
    z_col = _csv_column(columns, "axial-coordinate", "z-coordinate", "z")
    if any(item is None for item in [pressure_col, z_col]):
        raise ValueError(f"CSV {csv_file} is missing one or more required flow-field columns.")

    radius, theta, x_local, y_local, xy_center, radius_source, geometry_diagnostics = _infer_cylindrical_frame_from_xyz(
        df,
        rh=config.rh,
        rs=config.rs,
    )
    ux, uy, uz_xyz, velocity_source = _cartesian_velocity_from_csv(df, columns)
    ur_values, ut_values, uz_values = _cylindrical_velocity_from_cartesian(
        ux,
        uy,
        uz_xyz,
        radius,
        x_local,
        y_local,
    )
    if verbose_geometry_check:
        print(
            "CSV cylinder check: local cylindrical frame "
            f"r[q01,q99]=({geometry_diagnostics['local_radius_q01']:.6g}, "
            f"{geometry_diagnostics['local_radius_q99']:.6g}), "
            f"design=({config.rh:.6g}, {config.rs:.6g}), "
            f"axis_xy=({xy_center[0]:.6g}, {xy_center[1]:.6g}), "
            f"frame={radius_source}, velocity={velocity_source}. "
            "q01/q99 only describe sampled coverage, not exact wall radii."
        )

    z_physical = df[z_col].to_numpy(dtype=float)
    radial_tol = 0.02 * max(config.delta_r, 1e-12)
    sector_start = sector_index * theta0
    sector_theta = np.mod(theta - sector_start, 2.0 * np.pi)
    sector_mask = (
        (radius >= config.rh - radial_tol)
        & (radius <= config.rs + radial_tol)
        & (sector_theta >= 0.0)
        & (sector_theta <= theta0)
    )
    if int(np.sum(sector_mask)) < 128:
        raise ValueError(
            f"Only {int(np.sum(sector_mask))} CSV points found in sector {sector_index}; "
            "check the coordinate origin or theta_sector_index."
        )

    z_sector = z_physical[sector_mask]
    z_min = float(np.nanmin(z_sector))
    z_max = float(np.nanmax(z_sector))
    z_span = max(z_max - z_min, 1e-12)
    points = np.column_stack(
        [
            np.clip((radius[sector_mask] - config.rh) / max(config.delta_r, 1e-12), 0.0, 1.0),
            np.clip(sector_theta[sector_mask] / theta0, 0.0, 1.0),
            np.clip((z_sector - z_min) / z_span, 0.0, 1.0),
        ]
    )
    field_values = {
        "UR": ur_values[sector_mask],
        "UT": ut_values[sector_mask],
        "UZ": uz_values[sector_mask],
        "P": df[pressure_col].to_numpy(dtype=float)[sector_mask],
    }
    return {
        "points": points,
        "field_values": field_values,
        "csv_file": str(csv_file),
        "blade_params": str(blade_params),
        "theta_sector_index": sector_index,
        "points_in_sector": int(np.sum(sector_mask)),
        "xy_center": tuple(float(v) for v in xy_center),
        "radius_source": radius_source,
        "velocity_source": velocity_source,
        "z_min": z_min,
        "z_max": z_max,
        "geometry_diagnostics": geometry_diagnostics,
    }


def _physical_target_fields_from_case(
    case: Mapping[str, Any],
    *,
    pressure_reference: str = "training_origin",
) -> dict[str, np.ndarray]:
    # 把 case 中的 CFD 标签统一还原为柱坐标 SI 单位。
    # pressure_reference="training_origin" 与训练时的压力参考完全一致：P <- P - P[0,0,0]。
    config = FlowCaseConfig.from_mapping(case)
    required = ["UR", "UT", "UZ", "P"]
    if any(_pick(case, name) is None for name in required):
        raise ValueError("This case does not contain supervised CFD fields.")

    fields = {name: np.asarray(_pick(case, name), dtype=np.float32).copy() for name in required}
    fields_are_dimensionless = bool(_pick(case, "fields_are_dimensionless", default=True))
    if fields_are_dimensionless:
        fields["UR"] *= config.u_omega
        fields["UT"] *= config.u_omega
        fields["UZ"] *= config.u_zo
        fields["P"] *= config.P0

    if pressure_reference == "training_origin":
        fields["P"] = fields["P"] - float(fields["P"][0, 0, 0])
    elif pressure_reference == "absolute":
        pass
    else:
        raise ValueError("pressure_reference must be 'training_origin' or 'absolute'.")
    return fields


def _cfd_span_fields_from_case(
    case: Mapping[str, Any],
    *,
    span: float,
    n: int | None = None,
    interpolation_chunk_size: int = 250_000,
    pressure_reference: str = "training_origin",
) -> dict[str, np.ndarray]:
    # 优先从原始 CSV 直接插到指定 span 平面；若没有 CSV，就退回到 case 内已有网格标签的 R 向插值。
    n_out = int(n if n is not None else _pick(case, "n"))
    if _pick(case, "simulation_csv") is not None and _pick(case, "blade_params") is not None:
        scatter = _load_cfd_passage_scatter_from_case(case)
        fields = _interpolate_scattered_fields_to_span(
            scatter["points"],
            scatter["field_values"],
            span=span,
            n=n_out,
            chunk_size=interpolation_chunk_size,
        )
        if pressure_reference == "training_origin":
            reference = _physical_target_fields_from_case(case, pressure_reference="absolute")["P"]
            fields["P"] = fields["P"] - float(reference[0, 0, 0])
        elif pressure_reference != "absolute":
            raise ValueError("pressure_reference must be 'training_origin' or 'absolute'.")
        return {name: np.asarray(value, dtype=np.float32) for name, value in fields.items()}

    fields_grid = _physical_target_fields_from_case(case, pressure_reference=pressure_reference)
    return {name: _span_slice_from_grid(value, span) for name, value in fields_grid.items()}
