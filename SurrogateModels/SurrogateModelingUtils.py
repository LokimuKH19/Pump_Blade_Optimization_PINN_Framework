from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyvista as pv
import torch


def expand_scalar(x: torch.Tensor) -> torch.Tensor:
    # 把 batch 级标量扩成 [B, 1, 1, 1]，便于和三维场直接广播。
    return x.view(-1, 1, 1, 1)


def line_quadrature_weight(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 当前 R / Theta 网格都包含端点，因此线积分使用梯形权重。
    weight = torch.ones(length, device=device, dtype=dtype)
    if length > 1:
        weight[0] = 0.5
        weight[-1] = 0.5
    return weight


def neighbor_plus(x: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
    if periodic:
        return torch.roll(x, shifts=-1, dims=dim)
    out = torch.roll(x, shifts=-1, dims=dim)
    index = [slice(None)] * x.ndim
    index[dim] = -1
    out[tuple(index)] = x[tuple(index)]
    return out


def neighbor_minus(x: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
    if periodic:
        return torch.roll(x, shifts=1, dims=dim)
    out = torch.roll(x, shifts=1, dims=dim)
    index = [slice(None)] * x.ndim
    index[dim] = 0
    out[tuple(index)] = x[tuple(index)]
    return out


def d1_periodic_with_overlap(
    x: torch.Tensor,
    dim: int,
    spacing: torch.Tensor,
) -> torch.Tensor:
    # Theta 网格首尾是同一个物理位置，因此 seam 位置要按“去掉重复端点”的方式算导数。
    x_perm = torch.movedim(x, dim, -1)
    out = torch.zeros_like(x_perm)
    n = x_perm.shape[-1]
    spacing_value = spacing
    while spacing_value.ndim > x_perm.ndim - 1:
        spacing_value = spacing_value.squeeze(-1)

    if n <= 1:
        return torch.movedim(out, -1, dim)

    if n == 2:
        seam = (x_perm[..., 1] - x_perm[..., 0]) / torch.clamp(spacing_value, min=1e-12)
        out[..., 0] = seam
        out[..., 1] = seam
        return torch.movedim(out, -1, dim)

    out[..., 1:-1] = (x_perm[..., 2:] - x_perm[..., :-2]) / (2.0 * spacing)
    seam = (x_perm[..., 1] - x_perm[..., -2]) / (2.0 * spacing_value)
    out[..., 0] = seam
    out[..., -1] = seam
    return torch.movedim(out, -1, dim)


def d2_periodic_with_overlap(
    x: torch.Tensor,
    dim: int,
    spacing: torch.Tensor,
) -> torch.Tensor:
    x_perm = torch.movedim(x, dim, -1)
    out = torch.zeros_like(x_perm)
    n = x_perm.shape[-1]
    spacing_value = spacing
    while spacing_value.ndim > x_perm.ndim - 1:
        spacing_value = spacing_value.squeeze(-1)

    if n <= 1:
        return torch.movedim(out, -1, dim)

    if n == 2:
        seam = (x_perm[..., 1] - 2.0 * x_perm[..., 0] + x_perm[..., 1]) / (spacing_value ** 2)
        out[..., 0] = seam
        out[..., 1] = seam
        return torch.movedim(out, -1, dim)

    out[..., 1:-1] = (x_perm[..., 2:] - 2.0 * x_perm[..., 1:-1] + x_perm[..., :-2]) / (spacing ** 2)
    seam = (x_perm[..., 1] - 2.0 * x_perm[..., 0] + x_perm[..., -2]) / (spacing_value ** 2)
    out[..., 0] = seam
    out[..., -1] = seam
    return torch.movedim(out, -1, dim)


def field_stats(field: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    values = field[mask]
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(values.mean().item()),
        float(values.min().item()),
        float(values.max().item()),
    )


def case_summary(case: Mapping[str, Any]) -> dict[str, Any]:
    # checkpoint 里不保存整份大字段，只保留部署时真正需要核对的摘要。
    summary: dict[str, Any] = {}
    keys = [
        "n",
        "rh",
        "rs",
        "h",
        "mu",
        "rho",
        "omega",
        "qv",
        "n_blade",
        "n_blades",
        "z0",
        "g_star",
        "ibm_C",
        "ibm_epsilon",
        "absolute_frame",
        "blade_params",
    ]
    for key in keys:
        if key in case:
            value = case[key]
            if isinstance(value, Path):
                summary[key] = str(value)
            elif isinstance(value, (int, float, bool, str)):
                summary[key] = value
    return summary


def mask_surface(mask: np.ndarray) -> np.ndarray:
    # 只取叶片体素的外表面，减少三维散点数量，让转子轮廓更清楚。
    mask = np.asarray(mask, dtype=bool)

    r_plus = np.zeros_like(mask, dtype=bool)
    r_minus = np.zeros_like(mask, dtype=bool)
    z_plus = np.zeros_like(mask, dtype=bool)
    z_minus = np.zeros_like(mask, dtype=bool)

    r_plus[:-1, :, :] = mask[1:, :, :]
    r_minus[1:, :, :] = mask[:-1, :, :]
    z_plus[:, :, :-1] = mask[:, :, 1:]
    z_minus[:, :, 1:] = mask[:, :, :-1]

    theta_plus = np.roll(mask, shift=-1, axis=1)
    theta_minus = np.roll(mask, shift=1, axis=1)

    interior = mask & r_plus & r_minus & z_plus & z_minus & theta_plus & theta_minus
    return mask & (~interior)


def interpolate_field_periodic(
    field: np.ndarray,
    r_norm: float,
    theta_norm: float,
    z_norm: float,
) -> float:
    # 在带 Theta 周期性的三维标量场上做三线性插值。
    nr, nt, nz = field.shape

    r_norm = float(r_norm)
    theta_norm = float(theta_norm % 1.0)
    z_norm = float(z_norm)

    if r_norm < 0.0 or r_norm > 1.0 or z_norm < 0.0 or z_norm > 1.0:
        return float("nan")

    r_pos = np.clip(r_norm * (nr - 1), 0.0, nr - 1 - 1e-8)
    z_pos = np.clip(z_norm * (nz - 1), 0.0, nz - 1 - 1e-8)

    if nt <= 1:
        theta_pos = 0.0
    else:
        theta_pos = (theta_norm * (nt - 1)) % (nt - 1)

    i0 = int(np.floor(r_pos))
    j0 = int(np.floor(theta_pos))
    k0 = int(np.floor(z_pos))
    i1 = min(i0 + 1, nr - 1)
    j1 = min(j0 + 1, nt - 1)
    k1 = min(k0 + 1, nz - 1)

    ar = r_pos - i0
    at = theta_pos - j0
    az = z_pos - k0

    c000 = field[i0, j0, k0]
    c001 = field[i0, j0, k1]
    c010 = field[i0, j1, k0]
    c011 = field[i0, j1, k1]
    c100 = field[i1, j0, k0]
    c101 = field[i1, j0, k1]
    c110 = field[i1, j1, k0]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1.0 - ar) + c100 * ar
    c01 = c001 * (1.0 - ar) + c101 * ar
    c10 = c010 * (1.0 - ar) + c110 * ar
    c11 = c011 * (1.0 - ar) + c111 * ar
    c0 = c00 * (1.0 - at) + c10 * at
    c1 = c01 * (1.0 - at) + c11 * at
    return float(c0 * (1.0 - az) + c1 * az)


def build_pyvista_surface_mesh(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    valid_mask: np.ndarray,
) -> pv.PolyData | None:
    # 把规则网格上的叶片表面转成真正可渲染的三角面片。
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    nr, nt = z_grid.shape
    if nr < 2 or nt < 2:
        return None

    points = np.column_stack([x_grid.reshape(-1), y_grid.reshape(-1), z_grid.reshape(-1)])
    faces: list[int] = []

    def point_id(i: int, j: int) -> int:
        return i * nt + j

    for i in range(nr - 1):
        for j in range(nt - 1):
            corners = [
                valid_mask[i, j],
                valid_mask[i + 1, j],
                valid_mask[i, j + 1],
                valid_mask[i + 1, j + 1],
            ]
            if not all(corners):
                continue

            p00 = point_id(i, j)
            p10 = point_id(i + 1, j)
            p01 = point_id(i, j + 1)
            p11 = point_id(i + 1, j + 1)
            faces.extend([3, p00, p10, p11, 3, p00, p11, p01])

    if not faces:
        return None

    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    return mesh.clean()


def make_pyvista_passage_grid(
    config: Any,
    *,
    nr: int = 24,
    ntheta: int = 144,
    nz: int = 28,
) -> pv.StructuredGrid:
    # 生成整圈流道的参考网格，用作半透明背景。
    r = np.linspace(config.rh, config.rs, nr, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, dtype=float)
    z = np.linspace(config.z0, config.z0 + config.h, nz, dtype=float)
    r_grid, theta_grid, z_grid = np.meshgrid(r, theta, z, indexing="ij")
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    return pv.StructuredGrid(x_grid, y_grid, z_grid)


def make_pyvista_blade_surface_meshes(boundary: Any, config: Any) -> list[pv.PolyData]:
    # 按 blade_params 重建单流道叶片表面，并在周向复制成整圈转子。
    meshes: list[pv.PolyData] = []
    r_norm = np.asarray(boundary.r_coords, dtype=float)
    theta_norm = np.asarray(boundary.theta_coords, dtype=float)
    r_phys = config.rh + r_norm[:, None] * config.delta_r
    theta_base = theta_norm[None, :] * config.theta0
    valid_mask = np.asarray(boundary.footprint_mask, dtype=bool)

    z_upper = config.z0 + np.asarray(boundary.z_upper, dtype=float) * config.h
    z_lower = config.z0 + np.asarray(boundary.z_lower, dtype=float) * config.h

    for blade_id in range(config.n_blade):
        theta_phys = theta_base + blade_id * config.theta0
        x_grid = r_phys * np.cos(theta_phys)
        y_grid = r_phys * np.sin(theta_phys)

        upper_mesh = build_pyvista_surface_mesh(x_grid, y_grid, z_upper, valid_mask & np.isfinite(z_upper))
        lower_mesh = build_pyvista_surface_mesh(x_grid, y_grid, z_lower, valid_mask & np.isfinite(z_lower))

        if upper_mesh is not None:
            meshes.append(upper_mesh)
        if lower_mesh is not None:
            meshes.append(lower_mesh)

    return meshes
