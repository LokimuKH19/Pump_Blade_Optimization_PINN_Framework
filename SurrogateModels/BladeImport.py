# 用来边界识别的程序
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from DataGenerator import BladeCalc


# 根据控制点输入，绘制贝塞尔曲线
def bezier_curve(x: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    n = len(ctrl) - 1
    gamma = np.zeros_like(x, dtype=float)
    for i, value in enumerate(ctrl):
        gamma += math.comb(n, i) * (x ** i) * ((1.0 - x) ** (n - i)) * value
    max_value = float(np.max(gamma)) if gamma.size else 0.0
    if max_value > 1e-12:
        gamma /= max_value
    return gamma


# 厚度用的是三次样条插值
def spline_thickness(x: np.ndarray, knots_x: np.ndarray, knots_t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    knots_x = np.asarray(knots_x, dtype=float)
    knots_t = np.asarray(knots_t, dtype=float)
    tau = CubicSpline(knots_x, knots_t)(x)
    tau = np.clip(tau, 0.0, None)
    max_value = float(np.max(tau)) if tau.size else 0.0
    if max_value > 1e-12:
        tau /= max_value
    return tau


# 流道几何是用来归一化的参数集合
@dataclass(frozen=True)
class PassageGeometry:
    hub_radius: float
    shroud_radius: float
    passage_height: float
    blade_count: int
    grid_size: int
    passage_z0: float = 0.0

    @property
    def delta_r(self) -> float:
        return self.shroud_radius - self.hub_radius

    @property
    def sector_angle(self) -> float:
        return 2.0 * math.pi / self.blade_count

    @property
    def spacing(self) -> tuple[float, float, float]:
        if self.grid_size <= 1:
            return 1.0, 1.0, 1.0
        step = 1.0 / (self.grid_size - 1)
        return step, step, step

    @classmethod
    def from_solver(cls, solver: "BladeCalc") -> "PassageGeometry":
        return cls(
            hub_radius=float(solver.rh),
            shroud_radius=float(solver.rs),
            passage_height=float(solver.h),
            blade_count=int(solver.n_blade),
            grid_size=int(solver.n),
            passage_z0=float(getattr(solver, "z0", 0.0)),
        )


# 叶片几何
@dataclass
class BladeBoundaryData:
    mask: np.ndarray
    signed_distance: np.ndarray
    signed_distance_z: np.ndarray
    footprint_mask: np.ndarray
    z_lower: np.ndarray
    z_upper: np.ndarray
    r_coords: np.ndarray
    theta_coords: np.ndarray
    z_coords: np.ndarray
    metadata: dict[str, Any]

    # 把数据给字典化
    def as_numpy_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask,
            "signed_distance": self.signed_distance,
            "signed_distance_z": self.signed_distance_z,
            "footprint_mask": self.footprint_mask,
            "z_lower": self.z_lower,
            "z_upper": self.z_upper,
            "r_coords": self.r_coords,
            "theta_coords": self.theta_coords,
            "z_coords": self.z_coords,
            "metadata": self.metadata,
        }


class BladeImporter:
    def __init__(self, blade_params: str | Path | Mapping[str, Any], passage: PassageGeometry):
        self.raw = self._load_params(blade_params)
        self.passage = passage

        global_parameters = self.raw["global_parameters"]
        self.layers = list(self.raw["blade_layers"])
        self.generation_info = dict(self.raw.get("generation_info", {}))

        self.source_hub_radius = float(global_parameters["hub_radius"])
        self.source_shroud_radius = float(global_parameters["shroud_radius"])
        self.source_passage_height = float(global_parameters["passage_height_H1"])
        self.source_passage_z0 = float(global_parameters.get("z0", 0.0))
        self.source_blade_count = int(global_parameters["blade_count_N"])

        self.blade_height = float(global_parameters["blade_height_H"])
        self.blade_theta_span = float(global_parameters["Theta"])
        self.align = str(self.generation_info.get("align", "left")).lower()
        self.theta_offset = self._resolve_theta_offset()
        self.blade_z0 = self._resolve_blade_z0()

        self._build_interpolators()

    # 从json文件中读取叶片
    @staticmethod
    def _load_params(blade_params: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
        if isinstance(blade_params, Mapping):
            return blade_params
        path = Path(blade_params)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    # 读取theta的偏移值，每层
    def _resolve_theta_offset(self) -> float:
        if "theta_offset_rad" in self.generation_info:
            return float(self.generation_info["theta_offset_rad"])
        if self.align == "center":
            return 0.5 * (self.passage.sector_angle - self.blade_theta_span)
        return 0.0

    # 读取z偏置
    def _resolve_blade_z0(self) -> float:
        return self.passage.passage_z0 + 0.5 * (self.passage.passage_height - self.blade_height)

    # 对每层叶片进行插值
    def _build_interpolators(self) -> None:
        layer_count = len(self.layers)
        if layer_count < 2:
            raise ValueError("BladeImport requires at least two spanwise layers.")

        s = np.linspace(0.0, 1.0, layer_count)

        def collect(name: str) -> np.ndarray:
            return np.array([float(layer[name]) for layer in self.layers], dtype=float)

        self._theta0_s = PchipInterpolator(s, collect("theta0"))
        self._hmax_s = PchipInterpolator(s, collect("h_max"))
        self._tmax_s = PchipInterpolator(s, collect("t_max"))

        radius_values = []
        for index, layer in enumerate(self.layers):
            if "radius" in layer:
                radius_values.append(float(layer["radius"]))
            else:
                ratio = index / (layer_count - 1)
                radius_values.append(
                    (1.0 - ratio) * self.source_hub_radius + ratio * self.source_shroud_radius
                )
        self._radius_s = PchipInterpolator(s, np.asarray(radius_values, dtype=float))

        camber_layers = np.array([layer["camber_ctrl"] for layer in self.layers], dtype=float)
        thickness_layers = np.array(
            [layer["thickness_knots"]["t"] for layer in self.layers],
            dtype=float,
        )
        self._thickness_knots_x = np.asarray(self.layers[0]["thickness_knots"]["x"], dtype=float)
        self._camber_ctrl_s = [PchipInterpolator(s, camber_layers[:, i]) for i in range(camber_layers.shape[1])]
        self._thickness_ctrl_s = [
            PchipInterpolator(s, thickness_layers[:, i])
            for i in range(thickness_layers.shape[1])
        ]

        self._build_radius_inverse()

    # r向插值器构建，以定义各层
    def _build_radius_inverse(self) -> None:
        s_dense = np.linspace(0.0, 1.0, 2049)
        r_dense = np.asarray(self._radius_s(s_dense), dtype=float)
        dr = np.diff(r_dense)
        tol = 1e-10

        if np.all(dr >= -tol):
            r_monotonic = r_dense
            s_monotonic = s_dense
        elif np.all(dr <= tol):
            r_monotonic = r_dense[::-1]
            s_monotonic = s_dense[::-1]
        else:
            raise ValueError("Blade layer radii must stay monotonic from hub to shroud.")

        keep = np.ones_like(r_monotonic, dtype=bool)
        keep[1:] = np.abs(np.diff(r_monotonic)) > 1e-12
        r_unique = r_monotonic[keep]
        s_unique = s_monotonic[keep]
        if r_unique.size < 2:
            raise ValueError("Blade radius definition collapsed to a single value.")

        self.radius_min = float(r_unique[0])
        self.radius_max = float(r_unique[-1])
        self._span_from_radius = PchipInterpolator(r_unique, s_unique, extrapolate=False)

    # 每层的型线
    def _surface_curves(self, s_value: float, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_values = np.asarray(x_values, dtype=float)
        camber_ctrl = np.array([fn(s_value) for fn in self._camber_ctrl_s], dtype=float)
        thickness_ctrl = np.array([fn(s_value) for fn in self._thickness_ctrl_s], dtype=float)

        gamma = bezier_curve(x_values, camber_ctrl)
        tau = spline_thickness(x_values, self._thickness_knots_x, thickness_ctrl)

        h_max = float(self._hmax_s(s_value))
        t_max = float(self._tmax_s(s_value))

        z_center = self.blade_z0 + x_values * self.blade_height - h_max * gamma
        z_upper = z_center + t_max * tau
        z_lower = z_center - t_max * tau
        return z_lower, z_upper

    # 做成标准化网格的形式，1*1*1, 按照每层来做
    def rasterize(self) -> BladeBoundaryData:
        n = self.passage.grid_size
        r_coords = np.linspace(0.0, 1.0, n, dtype=float)
        theta_coords = np.linspace(0.0, 1.0, n, dtype=float)
        z_coords = np.linspace(0.0, 1.0, n, dtype=float)

        theta_line_phys = theta_coords * self.passage.sector_angle
        radius_line_phys = self.passage.hub_radius + r_coords * self.passage.delta_r

        footprint_mask = np.zeros((n, n), dtype=bool)
        z_lower = np.full((n, n), np.nan, dtype=np.float32)
        z_upper = np.full((n, n), np.nan, dtype=np.float32)

        theta_pad = 0.5 * self.passage.spacing[1] * self.passage.sector_angle

        for i, radius in enumerate(radius_line_phys):
            if radius < self.radius_min - 1e-12 or radius > self.radius_max + 1e-12:
                continue

            s_value = float(self._span_from_radius(radius))
            if not math.isfinite(s_value):
                continue

            theta_le = self.theta_offset + float(self._theta0_s(s_value))
            theta_te = theta_le + self.blade_theta_span
            theta_inside = (theta_line_phys >= theta_le - theta_pad) & (theta_line_phys <= theta_te + theta_pad)
            if not np.any(theta_inside):
                continue

            x_values = (theta_line_phys[theta_inside] - theta_le) / self.blade_theta_span
            x_values = np.clip(x_values, 0.0, 1.0)
            z_lower_phys, z_upper_phys = self._surface_curves(s_value, x_values)

            footprint_mask[i, theta_inside] = True
            z_lower[i, theta_inside] = (
                (z_lower_phys - self.passage.passage_z0) / self.passage.passage_height
            ).astype(np.float32)
            z_upper[i, theta_inside] = (
                (z_upper_phys - self.passage.passage_z0) / self.passage.passage_height
            ).astype(np.float32)

        z_grid = z_coords[None, None, :]
        footprint_3d = footprint_mask[:, :, None]
        z_lower_3d = z_lower[:, :, None]
        z_upper_3d = z_upper[:, :, None]

        mask = footprint_3d & (z_grid >= z_lower_3d) & (z_grid <= z_upper_3d)
        mask = mask.astype(bool)

        signed_distance_z = np.full(mask.shape, np.inf, dtype=np.float32)
        lower_broadcast = np.broadcast_to(z_lower_3d, mask.shape)
        upper_broadcast = np.broadcast_to(z_upper_3d, mask.shape)
        z_broadcast = np.broadcast_to(z_grid, mask.shape)
        footprint_broadcast = np.broadcast_to(footprint_3d, mask.shape)    # 揭示边界影响区域，但感觉没什么必要了

        below = footprint_broadcast & (z_broadcast < lower_broadcast)
        above = footprint_broadcast & (z_broadcast > upper_broadcast)

        signed_distance_z[below] = (lower_broadcast - z_broadcast)[below].astype(np.float32)
        signed_distance_z[above] = (z_broadcast - upper_broadcast)[above].astype(np.float32)

        inside_distance = np.minimum(
            (z_broadcast - lower_broadcast),
            (upper_broadcast - z_broadcast),
        )
        signed_distance_z[mask] = (-inside_distance[mask]).astype(np.float32)

        # 计算欧氏距离，支持 theta 方向周期边界（左右各扩展一个完整扇区）
        n_r, n_theta, n_z = mask.shape
        # 扩展：左周期副本 + 主扇区 + 右周期副本
        mask_ext = np.concatenate([mask[:, -n_theta:, :],   # 左邻扇区（取末尾 n_theta 层）
                                   mask,
                                   mask[:, :n_theta, :]], axis=1)   # 右邻扇区（取开头 n_theta 层）

        spacing = self.passage.spacing  # (dr_step, dtheta_step, dz_step)
        # 在扩展网格上计算距离（外部为正，内部为负）
        outside_ext = distance_transform_edt(~mask_ext, sampling=spacing)
        inside_ext = distance_transform_edt(mask_ext, sampling=spacing)
        signed_distance_ext = outside_ext.astype(np.float32)
        signed_distance_ext[mask_ext] = -inside_ext[mask_ext].astype(np.float32)

        # 裁剪回主扇区（中间部分，索引 n_theta : 2*n_theta）
        signed_distance = signed_distance_ext[:, n_theta:2*n_theta, :].copy()

        # 将主扇区内部点距离强制为 0
        signed_distance[mask] = 0.0

        metadata = {
            "align": self.align,
            "theta_offset_rad": self.theta_offset,
            "blade_z0_physical": self.blade_z0,
            "source_geometry": {
                "hub_radius": self.source_hub_radius,
                "shroud_radius": self.source_shroud_radius,
                "passage_height": self.source_passage_height,
                "passage_z0": self.source_passage_z0,
                "blade_count": self.source_blade_count,
            },
            "target_geometry": {
                "hub_radius": self.passage.hub_radius,
                "shroud_radius": self.passage.shroud_radius,
                "passage_height": self.passage.passage_height,
                "passage_z0": self.passage.passage_z0,
                "blade_count": self.passage.blade_count,
                "grid_size": self.passage.grid_size,
            },
            "consistency_notes": self._build_consistency_notes(),
            "mask_points": int(mask.sum()),
        }

        return BladeBoundaryData(
            mask=mask,
            signed_distance=signed_distance,
            signed_distance_z=signed_distance_z,
            footprint_mask=footprint_mask,
            z_lower=z_lower,
            z_upper=z_upper,
            r_coords=r_coords.astype(np.float32),
            theta_coords=theta_coords.astype(np.float32),
            z_coords=z_coords.astype(np.float32),
            metadata=metadata,
        )

    # 检查几何
    def _build_consistency_notes(self) -> list[str]:
        notes: list[str] = []
        if abs(self.source_hub_radius - self.passage.hub_radius) > 1e-9:
            notes.append("Target hub radius differs from blade_params.json and is used for normalization.")
        if abs(self.source_shroud_radius - self.passage.shroud_radius) > 1e-9:
            notes.append("Target shroud radius differs from blade_params.json and is used for normalization.")
        if abs(self.source_passage_height - self.passage.passage_height) > 1e-9:
            notes.append("Target passage height differs from blade_params.json and is used for normalization.")
        if self.source_blade_count != self.passage.blade_count:
            notes.append("Target blade count differs from blade_params.json; sector angle comes from BladeCalc.")
        if self.blade_theta_span > self.passage.sector_angle + 1e-12:
            notes.append("Blade chord theta span is larger than the target sector angle.")
        return notes


def build_blade_boundary(
    blade_params: str | Path | Mapping[str, Any],
    passage: PassageGeometry | "BladeCalc",
) -> BladeBoundaryData:
    passage_geometry = passage if isinstance(passage, PassageGeometry) else PassageGeometry.from_solver(passage)
    return BladeImporter(blade_params, passage_geometry).rasterize()


def attach_blade_to_solver(
    solver: "BladeCalc",
    blade_params: str | Path | Mapping[str, Any],
    *,
    band_cells: float = 1.5,
) -> BladeBoundaryData:
    boundary = build_blade_boundary(blade_params, solver)
    solver.set_blade_boundary(boundary, band_cells=band_cells)
    return boundary


# 测试效果
def plot_boundary_debug(
    boundary: BladeBoundaryData,
    mode: str = "rtheta",
    slice_index: int | None = None,
    detect_periodicity: bool = True,
) -> None:
    """
    强一致性调试可视化（绝不混轴版本）

    坐标约定（严格）：
        axis=0 → r（纵轴）
        axis=1 → θ（横轴）
        axis=2 → z

    Parameters
    ----------
    mode : str
        'rtheta' : 固定 z，看 (r, θ)
        'thetaz' : 固定 r，看 (θ, z)
    """

    mask = boundary.mask
    dist = boundary.signed_distance

    r = boundary.r_coords
    theta = boundary.theta_coords
    z = boundary.z_coords

    # ========= 周期性检测 =========
    def check_periodicity(arr, axis):
        """检测某一方向是否周期"""
        if axis == 1:  # theta
            diff = np.nanmean(np.abs(arr[:, 0, :] - arr[:, -1, :]))
        elif axis == 2:  # z
            diff = np.nanmean(np.abs(arr[:, :, 0] - arr[:, :, -1]))
        else:
            return False, None

        return diff < 1e-2, diff

    if detect_periodicity:
        theta_periodic, theta_err = check_periodicity(dist, axis=1)
        z_periodic, z_err = check_periodicity(dist, axis=2)

        print("\n[周期性检测]")
        print(f"theta方向: {'是' if theta_periodic else '否'} (误差={theta_err:.3e})")
        print(f"z方向    : {'是' if z_periodic else '否'} (误差={z_err:.3e})")

    # ========= r-theta 切片 =========
    if mode == "rtheta":

        if slice_index is None:
            slice_index = mask.shape[2] // 2

        z_idx = slice_index

        mask_slice = mask[:, :, z_idx]          # (r, theta)
        dist_slice = dist[:, :, z_idx]

        dist_slice = dist_slice.copy()
        dist_slice[np.isinf(dist_slice)] = np.nan

        inside_vals = dist_slice[mask_slice]
        if inside_vals.size > 0:
            print(
                f"[rtheta] z_idx={z_idx}, 内部距离: "
                f"min={inside_vals.min():.3e}, "
                f"max={inside_vals.max():.3e}, "
                f"mean={inside_vals.mean():.3e}"
            )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        extent = [theta.min(), theta.max(), r.min(), r.max()]

        # footprint
        axes[0].imshow(
            boundary.footprint_mask,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="gray_r",
        )
        axes[0].set_title("Footprint (r-θ)")
        axes[0].set_xlabel("θ")
        axes[0].set_ylabel("r")

        # mask
        axes[1].imshow(
            mask_slice,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="gray_r",
        )
        axes[1].set_title(f"Mask @ z={z[z_idx]:.3f}")
        axes[1].set_xlabel("θ")
        axes[1].set_ylabel("r")

        # distance
        im = axes[2].imshow(
            dist_slice,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
        )
        axes[2].set_title("Signed Distance")
        axes[2].set_xlabel("θ")
        axes[2].set_ylabel("r")
        fig.colorbar(im, ax=axes[2])

    # ========= theta-z 切片 =========
    elif mode == "thetaz":

        if slice_index is None:
            slice_index = mask.shape[0] // 2

        r_idx = slice_index

        mask_slice = mask[r_idx, :, :]          # (theta, z)
        dist_slice = dist[r_idx, :, :]          # (theta, z)

        dist_slice = dist_slice.copy()
        dist_slice[np.isinf(dist_slice)] = np.nan

        inside_vals = dist_slice[mask_slice]
        if inside_vals.size > 0:
            print(
                f"[thetaz] r_idx={r_idx}, 内部距离: "
                f"min={inside_vals.min():.3e}, "
                f"max={inside_vals.max():.3e}, "
                f"mean={inside_vals.mean():.3e}"
            )

        # 防止混轴，这里要转置
        mask_plot = mask_slice.T      # (z, theta)
        dist_plot = dist_slice.T

        extent = [theta.min(), theta.max(), z.min(), z.max()]

        z_low = boundary.z_lower[r_idx, :]
        z_up = boundary.z_upper[r_idx, :]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 轮廓
        ax = axes[0]
        valid = ~np.isnan(z_low)
        if np.any(valid):
            ax.fill_between(theta[valid], z_low[valid], z_up[valid], alpha=0.3)
            ax.plot(theta[valid], z_low[valid], linewidth=1)
            ax.plot(theta[valid], z_up[valid], linewidth=1)

        ax.set_title(f"Blade Profile @ r={r[r_idx]:.3f}")
        ax.set_xlabel("θ")
        ax.set_ylabel("z")
        ax.grid(True, linestyle=":")

        # mask
        axes[1].imshow(
            mask_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="gray_r",
        )
        axes[1].set_title("Mask (θ-z)")
        axes[1].set_xlabel("θ")
        axes[1].set_ylabel("z")

        # distance（限制颜色范围避免误判）
        max_ext = np.nanmax(dist_plot[dist_plot > 0]) if np.any(dist_plot > 0) else 1.0

        im = axes[2].imshow(
            dist_plot,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=0,
            vmax=max_ext,
        )
        axes[2].set_title("Signed Distance")
        axes[2].set_xlabel("θ")
        axes[2].set_ylabel("z")
        fig.colorbar(im, ax=axes[2])

    else:
        raise ValueError("mode must be 'rtheta' or 'thetaz'")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    params_path = here / "../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc/blade_params.json"
    with params_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    global_params = params["global_parameters"]
    passage = PassageGeometry(
        hub_radius=float(global_params["hub_radius"]),
        shroud_radius=float(global_params["shroud_radius"]),
        passage_height=float(global_params["passage_height_H1"]),
        blade_count=int(global_params["blade_count_N"]),
        grid_size=256,
        passage_z0=float(global_params.get("z0", 0.0)),
    )
    boundary = build_blade_boundary(params, passage)
    print(f"Blade mask points: {boundary.metadata['mask_points']}")
    for note in boundary.metadata["consistency_notes"]:
        print(f"- {note}")
    plot_boundary_debug(boundary, mode='thetaz')

