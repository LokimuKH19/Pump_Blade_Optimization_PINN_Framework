from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

import NeuralOperators
from SurrogateModelingConfig import FlowCaseConfig
from SurrogateModelingData import _cfd_span_fields_from_case, _physical_target_fields_from_case, _span_slice_from_grid
from SurrogateModelingUtils import (
    _profile_mean_min_max,
    field_stats,
    interpolate_field_periodic,
    make_pyvista_blade_surface_meshes,
    make_pyvista_passage_grid,
    span_to_index,
)


def _trace_streamline_cylindrical(
    self,
    *,
    fields_phy: Mapping[str, np.ndarray],
    phi_field: np.ndarray,
    config: FlowCaseConfig,
    seed_r: float,
    seed_theta: float,
    seed_z: float,
    max_steps: int = 500,
    step_scale: float = 0.75,
    phi_stop: float = 0.25,
) -> np.ndarray:
    # 在圆柱坐标下积分流线，然后再映射到三维笛卡尔坐标中画图。
    # 暂时先确保稳定、可读、易调试，而不是特别高阶的积分精度。
    dr_cell = config.delta_r / max(config.n - 1, 1)
    dz_cell = config.h / max(config.n - 1, 1)
    dtheta_cell = config.rh * config.theta0 / max(config.n - 1, 1)
    step_length = step_scale * min(dr_cell, dz_cell, dtheta_cell)

    def eval_state(r_value: float, theta_value: float, z_value: float):
        r_norm = (r_value - config.rh) / config.delta_r
        theta_norm = (theta_value % config.theta0) / config.theta0
        z_norm = (z_value - config.z0) / config.h

        phi_value = interpolate_field_periodic(phi_field, r_norm, theta_norm, z_norm)
        if not np.isfinite(phi_value) or phi_value < phi_stop:
            return None

        ur = interpolate_field_periodic(fields_phy["UR"], r_norm, theta_norm, z_norm)
        ut = interpolate_field_periodic(fields_phy["UT"], r_norm, theta_norm, z_norm)
        uz = interpolate_field_periodic(fields_phy["UZ"], r_norm, theta_norm, z_norm)
        if not np.isfinite(ur + ut + uz):
            return None

        speed = float(np.sqrt(ur ** 2 + ut ** 2 + uz ** 2))
        if speed < 1e-10:
            return None

        rhs = np.array([ur, ut / max(r_value, 1e-10), uz], dtype=float)
        return rhs, speed

    state = np.array([seed_r, seed_theta, seed_z], dtype=float)
    points: list[np.ndarray] = []

    for _ in range(max_steps):
        if state[0] < config.rh or state[0] > config.rs:
            break
        if state[2] < config.z0 or state[2] > config.z0 + config.h:
            break

        result = eval_state(float(state[0]), float(state[1]), float(state[2]))
        if result is None:
            break

        rhs_1, speed_1 = result
        dt = step_length / max(speed_1, 1e-10)
        mid_state = state + 0.5 * dt * rhs_1

        result_mid = eval_state(float(mid_state[0]), float(mid_state[1]), float(mid_state[2]))
        if result_mid is None:
            break

        rhs_2, _ = result_mid
        state = state + dt * rhs_2

        theta_plot = state[1] % config.theta0
        x_value = state[0] * np.cos(theta_plot)
        y_value = state[0] * np.sin(theta_plot)
        points.append(np.array([x_value, y_value, state[2]], dtype=float))

    if len(points) < 2:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(points)

def plot_3d_streamlines(
    self,
    *,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    seed_r_count: int = 10,
    seed_theta_count: int = 10,
    passages_to_plot: int | None = None,
    max_streamline_steps: int = 1000,
    streamline_step_scale: float = 0.9,
    theme: str = "light",
) -> dict[str, Any]:
    # 三维后处理改为 PyVista 风格，组织方式参考 BladeGeneratorCAD.py 里的可视化：
    # 1. 透明环形流道网格
    # 2. 转子上下表面实体
    # 3. hub / shroud 环线
    # 4. 三维流线
    bundle = self._predict_case_bundle(case_index=case_index, case=case)
    case_data = bundle["case"]
    sample = bundle["sample"]
    pred_phy = bundle["pred_phy"]
    boundary = bundle["boundary"]
    config = FlowCaseConfig.from_mapping(case_data)
    ibm_c_mean, ibm_c_min, ibm_c_max = _profile_mean_min_max(sample["ibm_C"])
    ibm_epsilon_mean, ibm_epsilon_min, ibm_epsilon_max = _profile_mean_min_max(sample["ibm_epsilon"])

    train_n = int(self.train_dataset[0]["x"].shape[-1])
    deploy_n = int(sample["x"].shape[-1])
    bg = "#0E1117" if theme == "dark" else "white"
    blade_color = "#FFB000" if theme == "dark" else "#C2410C"
    passage_color = "#A8A29E" if theme == "dark" else "#CBD5E1"
    hub_color = "#60A5FA" if theme == "dark" else "#2563EB"
    shroud_color = "#34D399" if theme == "dark" else "#059669"
    streamline_cmap = plt.cm.viridis if theme == "dark" else plt.cm.turbo
    plotter = pv.Plotter(off_screen=not show, window_size=(1400, 950))
    plotter.set_background(bg)

    passage_grid = make_pyvista_passage_grid(config)
    plotter.add_mesh(
        passage_grid,
        color=passage_color,
        opacity=0.22,
        show_edges=True,
        line_width=0.6,
    )

    if boundary is not None:
        for blade_mesh in make_pyvista_blade_surface_meshes(boundary, config):
            plotter.add_mesh(
                blade_mesh,
                color=blade_color,
                show_edges=False,
                smooth_shading=True,
                ambient=0.20,
                diffuse=0.85,
                specular=0.18,
            )

    # hub / shroud 环线沿用参考程序的组织方式，帮助观察叶片上下边界。
    theta_ring = np.linspace(0.0, 2.0 * np.pi, 240, dtype=float)
    z_root = config.z0
    z_tip = config.z0 + config.h
    for radius in [config.rh, config.rs]:
        x_ring = radius * np.cos(theta_ring)
        y_ring = radius * np.sin(theta_ring)
        root_line = np.column_stack([x_ring, y_ring, np.full_like(x_ring, z_root)])
        tip_line = np.column_stack([x_ring, y_ring, np.full_like(x_ring, z_tip)])
        plotter.add_lines(root_line, color=hub_color, width=2)
        plotter.add_lines(tip_line, color=shroud_color, width=2)

    # 入口种子点默认取 z 的第二层，避开边界本身。
    phi_inlet = sample["phi"][:, :, 1 if sample["phi"].shape[2] > 1 else 0].detach().cpu().numpy()
    r_coords = sample["R"][:, 0, 0].detach().cpu().numpy()
    theta_coords = sample["Theta"][0, :, 0].detach().cpu().numpy()
    z_seed = config.z0 + config.h * float(sample["Z"][0, 0, 1 if sample["Z"].shape[2] > 1 else 0].item())

    r_index_candidates = np.linspace(1, max(len(r_coords) - 2, 1), num=max(seed_r_count, 1), dtype=int)
    theta_index_candidates = np.linspace(1, max(len(theta_coords) - 2, 1), num=max(seed_theta_count, 1), dtype=int)

    base_seeds: list[tuple[float, float, float]] = []
    for i_index in r_index_candidates:
        for j_index in theta_index_candidates:
            if phi_inlet[i_index, j_index] > 0.75:
                seed_r = config.rh + float(r_coords[i_index]) * config.delta_r
                seed_theta = float(theta_coords[j_index]) * config.theta0
                base_seeds.append((seed_r, seed_theta, z_seed))

    if not base_seeds:
        fluid_indices = np.argwhere(phi_inlet > 0.75)
        for i_index, j_index in fluid_indices[:: max(1, len(fluid_indices) // 12)]:
            seed_r = config.rh + float(r_coords[i_index]) * config.delta_r
            seed_theta = float(theta_coords[j_index]) * config.theta0
            base_seeds.append((seed_r, seed_theta, z_seed))

    if passages_to_plot is None:
        passages_to_plot = config.n_blade
    passages_to_plot = max(1, min(passages_to_plot, config.n_blade))

    colors = streamline_cmap(np.linspace(0.08, 0.92, max(len(base_seeds), 1)))[:, :3]
    phi_field = sample["phi"].detach().cpu().numpy()
    fields_phy_np = {name: tensor.detach().cpu().numpy() for name, tensor in pred_phy.items()}
    tube_radius = 0.0075 * config.delta_r

    def rotate_about_z(points: np.ndarray, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        rotated = points.copy()
        x_value = points[:, 0].copy()
        y_value = points[:, 1].copy()
        rotated[:, 0] = c * x_value - s * y_value
        rotated[:, 1] = s * x_value + c * y_value
        return rotated

    streamline_count = 0
    base_streamline_count = 0
    for color, (seed_r, seed_theta, seed_z) in zip(colors, base_seeds):
        base_streamline = self._trace_streamline_cylindrical(
            fields_phy=fields_phy_np,
            phi_field=phi_field,
            config=config,
            seed_r=seed_r,
            seed_theta=seed_theta,
            seed_z=seed_z,
            max_steps=max_streamline_steps,
            step_scale=streamline_step_scale,
        )
        if base_streamline.shape[0] < 2:
            continue
        base_streamline_count += 1
        for blade_id in range(passages_to_plot):
            theta_shift = blade_id * config.theta0
            streamline = rotate_about_z(base_streamline, theta_shift)
            spline = pv.Spline(streamline, max(streamline.shape[0], 2))
            streamline_mesh = spline.tube(radius=tube_radius)
            plotter.add_mesh(
                streamline_mesh,
                color=tuple(float(c) for c in color),
                smooth_shading=True,
                opacity=0.92,
            )
            streamline_count += 1

    # 在画面左下角保留一点文字信息，直接体现“粗网格训练、细网格部署”的部署特征。
    plotter.add_text(
        f"Rotor Streamlines | train_n={train_n} | deploy_n={deploy_n} | "
        f"blades={config.n_blade} | C_mean={ibm_c_mean:.4g} [{ibm_c_min:.4g},{ibm_c_max:.4g}] | "
        f"eps_mean={ibm_epsilon_mean:.4g} [{ibm_epsilon_min:.4g},{ibm_epsilon_max:.4g}]",
        position="upper_left",
        font_size=10,
        color="white" if theme == "dark" else "black",
    )
    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.18)

    if show and save_path is not None:
        plotter.show(screenshot=str(save_path))
        print(f"三维流线图已保存到: {save_path}")
    elif show:
        plotter.show()
    else:
        if save_path is not None:
            plotter.screenshot(str(save_path))
            print(f"三维流线图已保存到: {save_path}")
        plotter.close()

    return {
        "streamline_count": streamline_count,
        "base_streamline_count": base_streamline_count,
        "streamline_copy_count": passages_to_plot,
        "train_n": train_n,
        "deploy_n": deploy_n,
        "boundary_available": boundary is not None,
        "renderer": "pyvista",
    }

def plot_blade_spans(
    self,
    case_index: int = 0,
    spans: Sequence[float] = (0.2, 0.5, 0.8),
    *,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    # 这个图首先用来确认叶片导入是否对齐。
    sample = self.train_dataset[case_index]
    batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
    batch = self._prepare_runtime_batch(batch)
    mask = sample["blade_mask"].detach().cpu().numpy()
    phi = batch["phi"][0].detach().cpu().numpy()
    signed_distance = sample["signed_distance"].detach().cpu().numpy()
    n = mask.shape[0]

    fig, axes = plt.subplots(len(spans), 3, figsize=(13, 3.6 * len(spans)), squeeze=False)
    for row, span in enumerate(spans):
        r_index = span_to_index(span, n)

        im0 = axes[row, 0].imshow(mask[r_index].T, origin="lower", aspect="auto", cmap="gray_r")
        axes[row, 0].set_title(f"Blade Mask @ span={span:.2f} (i={r_index})")
        axes[row, 0].set_xlabel("Theta index")
        axes[row, 0].set_ylabel("Z index")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(signed_distance[r_index].T, origin="lower", aspect="auto", cmap="coolwarm")
        axes[row, 1].set_title(f"Signed Distance @ span={span:.2f}")
        axes[row, 1].set_xlabel("Theta index")
        axes[row, 1].set_ylabel("Z index")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(phi[r_index].T, origin="lower", aspect="auto", cmap="GnBu")
        axes[row, 2].set_title(f"Phi @ span={span:.2f}")
        axes[row, 2].set_xlabel("Theta index")
        axes[row, 2].set_ylabel("Z index")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"叶片 span 调试图已保存到: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_training_history(
    self,
    history: Sequence[Mapping[str, float]],
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    history_plot_mode: str = "auto",
) -> None:
    # 训练损失曲线统一用对数纵轴来画。
    # 默认关心五条线：
    # 1. Data 损失
    # 2. 连续方程残差
    # 3. 径向动量残差
    # 4. 周向动量残差
    # 5. 轴向动量残差
    #
    # 如果当前数据集没有监督标签，就自动跳过 Data 曲线。
    if len(history) == 0:
        return

    has_supervised_target = any(sample["has_target"].item() > 0.5 for sample in self.train_dataset.samples)
    epochs = np.arange(1, len(history) + 1, dtype=float)

    def first_continuity_loss_scale() -> float | None:
        for item in history:
            for key in ("train_scaled_loss_reference", "train_loss_c"):
                value = item.get(key)
                if value is None:
                    continue
                value = float(value)
                if np.isfinite(value) and value > 0.0:
                    return value
        return None

    continuity_loss_scale = first_continuity_loss_scale()
    mode = str(history_plot_mode).lower()
    mode_aliases = {
        "loss": "raw",
        "mse": "raw",
        "raw_loss": "raw",
        "scaled": "scaled_loss",
        "normalized": "scaled_loss",
        "normalised": "scaled_loss",
        "residual": "scaled_residual",
        "rmse": "scaled_residual",
        "normalized_residual": "scaled_residual",
        "normalised_residual": "scaled_residual",
        "fluent": "fluent_scaled",
        "fluent_scaled_residual": "fluent_scaled",
    }
    mode = mode_aliases.get(mode, mode)
    allowed_modes = {"auto", "raw", "scaled_loss", "scaled_residual", "fluent_scaled", "both", "all"}
    if mode not in allowed_modes:
        raise ValueError(
            "history_plot_mode must be 'auto', 'raw', 'scaled_loss', "
            "'scaled_residual', 'fluent_scaled', 'all', or 'both'."
        )

    if mode == "auto":
        active_modes = ["raw", "scaled_loss", "scaled_residual"] if continuity_loss_scale is not None else ["raw"]
    elif mode in {"both", "all"}:
        active_modes = ["raw", "scaled_loss", "scaled_residual"] if continuity_loss_scale is not None else ["raw"]
    else:
        active_modes = [mode]
    if continuity_loss_scale is None:
        active_modes = ["raw" if item in {"scaled_loss", "scaled_residual"} else item for item in active_modes]
    active_modes = list(dict.fromkeys(active_modes))

    def values_for(prefix: str, kind: str, key: str) -> np.ndarray | None:
        if kind == "raw":
            full_key = f"{prefix}{key}"
            if not any(full_key in item for item in history):
                return None
            return np.array([float(item.get(full_key, np.nan)) for item in history], dtype=float)

        if kind == "scaled_loss":
            scaled_key = f"{prefix}scaled_{key}"
            if any(scaled_key in item for item in history):
                return np.array([float(item.get(scaled_key, np.nan)) for item in history], dtype=float)
            raw = values_for(prefix, "raw", key)
            if raw is None or continuity_loss_scale is None:
                return None
            return raw / max(float(continuity_loss_scale), 1e-30)

        if kind == "scaled_residual":
            name = key.removeprefix("loss_")
            scaled_key = f"{prefix}scaled_residual_{name}"
            if any(scaled_key in item for item in history):
                return np.array([float(item.get(scaled_key, np.nan)) for item in history], dtype=float)
            raw = values_for(prefix, "raw", key)
            if raw is None or continuity_loss_scale is None:
                return None
            return np.sqrt(np.maximum(raw, 0.0) / max(float(continuity_loss_scale), 1e-30))

        if kind == "fluent_scaled":
            legacy_key = f"{prefix}scaled_res_{key.removeprefix('loss_')}"
            if not any(legacy_key in item for item in history):
                return None
            return np.array([float(item.get(legacy_key, np.nan)) for item in history], dtype=float)

        return None

    def curve_specs_for(kind: str) -> list[tuple[str, str]]:
        curve_specs: list[tuple[str, str]] = []
        if kind == "raw":
            if has_supervised_target:
                curve_specs.append(("loss_data", "Data loss"))
            curve_specs.extend(
                [
                    ("loss_c", "MSE R_c"),
                    ("loss_r", "MSE R_r"),
                    ("loss_theta", "MSE R_theta"),
                    ("loss_z", "MSE R_z"),
                ]
            )
        elif kind == "scaled_loss":
            if has_supervised_target:
                curve_specs.append(("loss_data", "Scaled data loss"))
            curve_specs.extend(
                [
                    ("loss_phys", "Scaled physics loss"),
                    ("loss_c", "Scaled loss R_c"),
                    ("loss_r", "Scaled loss R_r"),
                    ("loss_theta", "Scaled loss R_theta"),
                    ("loss_z", "Scaled loss R_z"),
                ]
            )
        elif kind == "scaled_residual":
            curve_specs.extend(
                [
                    ("loss_c", "Scaled RMSE R_c"),
                    ("loss_r", "Scaled RMSE R_r"),
                    ("loss_theta", "Scaled RMSE R_theta"),
                    ("loss_z", "Scaled RMSE R_z"),
                ]
            )
        elif kind == "fluent_scaled":
            curve_specs.extend(
                [
                    ("loss_c", "Fluent-like R_c"),
                    ("loss_r", "Fluent-like R_r"),
                    ("loss_theta", "Fluent-like R_theta"),
                    ("loss_z", "Fluent-like R_z"),
                ]
            )
        return curve_specs

    metric_titles = {
        "raw": "Raw Loss",
        "scaled_loss": "Scaled Loss",
        "scaled_residual": "Scaled Residual RMSE",
        "fluent_scaled": "Fluent-like Scaled Residual",
    }
    y_labels = {
        "raw": "Loss",
        "scaled_loss": "Loss / initial train continuity loss",
        "scaled_residual": "RMSE residual / initial train continuity RMSE",
        "fluent_scaled": "Fluent-like scaled residual",
    }
    suffixes = {
        "raw": "loss",
        "scaled_loss": "scaled_loss",
        "scaled_residual": "scaled_residual",
        "fluent_scaled": "fluent_scaled",
    }

    base_save_path = Path(save_path) if save_path is not None else None
    if base_save_path is not None:
        base_save_path.parent.mkdir(parents=True, exist_ok=True)

    for active_mode in active_modes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
        metric_title = metric_titles[active_mode]
        panel_specs = [
            ("train_", f"Training {metric_title} History"),
            ("val_", f"Validation {metric_title} History"),
        ]
        curve_specs = curve_specs_for(active_mode)

        for ax, (prefix, title) in zip(axes[0], panel_specs):
            plotted = False
            for key, label in curve_specs:
                values = values_for(prefix, active_mode, key)
                if values is None:
                    continue
                if not np.any(np.isfinite(values)):
                    continue

                # 对数坐标不能直接画 0，这里只在绘图时做极小截断。
                values = np.where(np.isfinite(values), np.maximum(values, 1e-30), np.nan)
                ax.plot(epochs, values, linewidth=1.6, label=label)
                plotted = True

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(y_labels[active_mode])
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
            if plotted:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No curves", transform=ax.transAxes, ha="center", va="center")

        fig.tight_layout()
        if base_save_path is not None:
            output_path = base_save_path
            if len(active_modes) > 1:
                output_path = base_save_path.with_name(
                    f"{base_save_path.stem}_{suffixes[active_mode]}{base_save_path.suffix}"
                )
            plt.savefig(str(output_path), dpi=180, bbox_inches="tight")
            print(f"{metric_title} 曲线已保存到: {output_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

def _dct_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_last = torch.movedim(x, dim, -1)
    coeff = NeuralOperators.dct_1d(x_last)
    return torch.movedim(coeff, -1, dim)

def _axis_spectrum_energy(
    field: torch.Tensor,
    weight: torch.Tensor,
    *,
    axis: int,
    periodic: bool,
    modes_cutoff: int,
) -> dict[str, Any]:
    if periodic and field.shape[axis] > 2:
        slicer = [slice(None)] * field.ndim
        slicer[axis] = slice(0, -1)
        field = field[tuple(slicer)]
        weight = weight[tuple(slicer)]

    weight = torch.clamp(weight, min=0.0)
    weight_sum = torch.clamp(torch.sum(weight), min=1e-12)
    mean = torch.sum(field * weight) / weight_sum
    signal = (field - mean) * torch.sqrt(weight)

    if periodic:
        coeff = torch.fft.rfft(signal, dim=axis, norm="ortho")
        coeff_energy = torch.abs(coeff) ** 2
    else:
        coeff = _dct_along(signal, axis)
        coeff_energy = coeff ** 2

    reduce_dims = tuple(dim for dim in range(coeff_energy.ndim) if dim != axis)
    energy = torch.sum(coeff_energy, dim=reduce_dims).detach().cpu().to(torch.float64)
    total = torch.sum(energy)
    if float(total.item()) <= 1e-30:
        energy_fraction = torch.zeros_like(energy)
        cumulative = torch.zeros_like(energy)
        high_ratio = 0.0
        k95 = 0
    else:
        energy_fraction = energy / total
        cumulative = torch.cumsum(energy_fraction, dim=0)
        keep = int(max(1, min(modes_cutoff, energy_fraction.numel())))
        high_ratio = float(torch.sum(energy_fraction[keep:]).item())
        k95 = int(torch.searchsorted(cumulative, torch.tensor(0.95, dtype=cumulative.dtype)).item()) + 1

    return {
        "energy_fraction": [float(v) for v in energy_fraction.tolist()],
        "cumulative": [float(v) for v in cumulative.tolist()],
        "high_ratio_after_modes": high_ratio,
        "k95": k95,
    }

def frequency_energy_diagnostics(
    self,
    *,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    fields: Sequence[str] = ("UR", "UT", "UZ", "P"),
    modes_cutoff: int | None = None,
) -> dict[str, Any]:
    bundle = self._predict_case_bundle(case_index=case_index, case=case)
    pred_dim = bundle["pred_dim"]
    phi = torch.clamp(bundle["sample"]["phi"].to(torch.float32), min=0.0, max=1.0)
    cutoff = int(modes_cutoff if modes_cutoff is not None else self.model_config.get("modes", 8))

    diagnostics: dict[str, Any] = {
        "modes_cutoff": cutoff,
        "operator_variant": self.model_config.get("operator_variant", "unknown"),
        "grid_shape": list(phi.shape),
        "fields": {},
    }
    axis_specs = [
        ("R", 0, False),
        ("Theta", 1, True),
        ("Z", 2, False),
    ]
    for field_name in fields:
        if field_name not in pred_dim:
            continue
        field = pred_dim[field_name].to(torch.float32)
        if field_name == "P":
            weight_sum = torch.clamp(torch.sum(phi), min=1e-12)
            field = field - torch.sum(field * phi) / weight_sum
        diagnostics["fields"][field_name] = {
            axis_name: self._axis_spectrum_energy(
                field,
                phi,
                axis=axis,
                periodic=periodic,
                modes_cutoff=cutoff,
            )
            for axis_name, axis, periodic in axis_specs
        }
    return diagnostics

def plot_frequency_energy_trends(
    self,
    *,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    fields: Sequence[str] = ("UR", "UT", "UZ", "P"),
    modes_cutoff: int | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    diagnostics = self.frequency_energy_diagnostics(
        case_index=case_index,
        case=case,
        fields=fields,
        modes_cutoff=modes_cutoff,
    )
    field_names = list(diagnostics["fields"].keys())
    axis_names = ["R", "Theta", "Z"]
    if not field_names:
        return diagnostics

    fig, axes = plt.subplots(
        len(field_names),
        len(axis_names),
        figsize=(4.8 * len(axis_names), 3.1 * len(field_names)),
        squeeze=False,
    )
    cutoff = int(diagnostics["modes_cutoff"])
    for row, field_name in enumerate(field_names):
        for col, axis_name in enumerate(axis_names):
            ax = axes[row, col]
            axis_data = diagnostics["fields"][field_name][axis_name]
            energy = np.asarray(axis_data["energy_fraction"], dtype=float)
            cumulative = np.asarray(axis_data["cumulative"], dtype=float)
            modes = np.arange(energy.shape[0], dtype=float)
            ax.semilogy(modes, np.maximum(energy, 1e-14), color="tab:blue", linewidth=1.3, label="mode energy")
            ax.set_xlabel("Mode index")
            ax.set_ylabel("Energy fraction", color="tab:blue")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax.grid(True, alpha=0.25)
            ax.axvline(max(cutoff - 1, 0), color="0.35", linestyle=":", linewidth=1.1)

            twin = ax.twinx()
            twin.plot(modes, cumulative, color="tab:orange", linewidth=1.4, label="cumulative")
            twin.set_ylim(0.0, 1.02)
            twin.set_ylabel("Cumulative", color="tab:orange")
            twin.tick_params(axis="y", labelcolor="tab:orange")

            ax.set_title(
                f"{field_name} {axis_name}: "
                f"K95={axis_data['k95']}, high={axis_data['high_ratio_after_modes']:.2e}"
            )

    fig.suptitle(
        f"Frequency Energy Trends ({diagnostics['operator_variant']}, cutoff modes={cutoff})",
        y=1.005,
    )
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=180, bbox_inches="tight")
        print(f"频率能量趋势图已保存到: {save_path}")
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"频率能量诊断 JSON 已保存到: {summary_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return diagnostics

def plot_cfd_spans(
    self,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    spans: Sequence[float] = (0.2, 0.5, 0.8),
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    pressure_reference: str = "absolute",
    interpolation_chunk_size: int = 250_000,
) -> None:
    # 只画 CFD/CSV 解，单位保持在物理空间：UR/UT/UZ 为 m/s，P 为 Pa。
    # 绘图必须忠实展示原始压力；若需要训练零点一致性，可通过 pressure_reference 单独选择。
    # 若请求的 span 不是 CSV 原始采样面，会用和训练导入完全相同的散点插值器投到该平面。
    case_data, sample = self._resolve_case_sample(case_index=case_index, case=case)
    n = int(sample["x"].shape[-1])
    mask = sample["blade_mask"].detach().cpu().numpy()
    field_names = ["UR", "UT", "UZ", "P"]
    p_label = "P [Pa]" if pressure_reference == "absolute" else "P rel. [Pa]"
    labels = {
        "UR": "UR [m/s]",
        "UT": "UT [m/s]",
        "UZ": "UZ [m/s]",
        "P": p_label,
    }
    cmaps = {"UR": "coolwarm", "UT": "coolwarm", "UZ": "viridis", "P": "plasma"}

    fig, axes = plt.subplots(len(spans), 4, figsize=(18, 3.6 * len(spans)), squeeze=False)
    for row, span in enumerate(spans):
        cfd_fields = _cfd_span_fields_from_case(
            case_data,
            span=float(span),
            n=n,
            interpolation_chunk_size=interpolation_chunk_size,
            pressure_reference=pressure_reference,
        )
        blade_mask = _span_slice_from_grid(mask, float(span)).T > 0.5
        for col, name in enumerate(field_names):
            data = np.ma.array(cfd_fields[name].T, mask=blade_mask)
            image = axes[row, col].imshow(data, origin="lower", aspect="auto", cmap=cmaps[name])
            axes[row, col].contour(blade_mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
            axes[row, col].set_title(f"CFD {labels[name]} @ span={span:.2f}")
            axes[row, col].set_xlabel("Theta index")
            axes[row, col].set_ylabel("Z index")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"CFD span 图已保存到: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

def _field_similarity_metrics(
    pred: np.ndarray,
    cfd: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    valid = valid & np.isfinite(pred) & np.isfinite(cfd)
    if not np.any(valid):
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "relative_l2": float("nan"),
            "nrmse_range": float("nan"),
            "cosine_similarity": float("nan"),
            "pearson_r": float("nan"),
            "r2": float("nan"),
            "similarity_score": float("nan"),
        }

    pred_v = np.asarray(pred[valid], dtype=float)
    cfd_v = np.asarray(cfd[valid], dtype=float)
    diff = pred_v - cfd_v
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    cfd_l2 = float(np.sqrt(np.mean(cfd_v ** 2)) + 1e-12)
    relative_l2 = float(np.sqrt(np.mean(diff ** 2)) / cfd_l2)
    cfd_range = float(np.nanpercentile(cfd_v, 99) - np.nanpercentile(cfd_v, 1))
    nrmse_range = float(rmse / max(cfd_range, 1e-12))

    pred_norm = float(np.linalg.norm(pred_v))
    cfd_norm = float(np.linalg.norm(cfd_v))
    cosine = float(np.dot(pred_v, cfd_v) / max(pred_norm * cfd_norm, 1e-12))
    if pred_v.size > 1 and np.std(pred_v) > 1e-12 and np.std(cfd_v) > 1e-12:
        pearson = float(np.corrcoef(pred_v, cfd_v)[0, 1])
    else:
        pearson = float("nan")
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((cfd_v - np.mean(cfd_v)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    similarity_score = float(np.clip(1.0 - relative_l2, 0.0, 1.0))

    return {
        "rmse": rmse,
        "mae": mae,
        "relative_l2": relative_l2,
        "nrmse_range": nrmse_range,
        "cosine_similarity": cosine,
        "pearson_r": pearson,
        "r2": r2,
        "similarity_score": similarity_score,
    }


def _flow_similarity_diagnostics(
    *,
    pred_phy: Mapping[str, torch.Tensor | np.ndarray],
    case: Mapping[str, Any],
    mask: np.ndarray,
    pressure_reference: str,
) -> dict[str, Any]:
    cfd_fields = _physical_target_fields_from_case(case, pressure_reference=pressure_reference)
    pred_fields = {name: np.asarray(value, dtype=np.float32) for name, value in pred_phy.items()}
    if pressure_reference == "training_origin":
        pred_fields["P"] = pred_fields["P"] - float(pred_fields["P"][0, 0, 0])
    elif pressure_reference != "absolute":
        raise ValueError("pressure_reference must be 'training_origin' or 'absolute'.")

    fluid = np.asarray(mask) < 0.5
    fields: dict[str, Any] = {}
    for name in ["UR", "UT", "UZ", "P"]:
        fields[name] = _field_similarity_metrics(pred_fields[name], cfd_fields[name], fluid)

    velocity_valid = fluid.copy()
    pred_components = []
    cfd_components = []
    for name in ["UR", "UT", "UZ"]:
        pred_value = pred_fields[name]
        cfd_value = cfd_fields[name]
        velocity_valid = velocity_valid & np.isfinite(pred_value) & np.isfinite(cfd_value)
        pred_components.append(pred_value)
        cfd_components.append(cfd_value)
    if np.any(velocity_valid):
        pred_speed = np.sqrt(sum(component ** 2 for component in pred_components))
        cfd_speed = np.sqrt(sum(component ** 2 for component in cfd_components))
        velocity_metrics = _field_similarity_metrics(pred_speed, cfd_speed, velocity_valid)
        vec_diff_sq = sum((pred - cfd) ** 2 for pred, cfd in zip(pred_components, cfd_components))
        vec_ref_sq = sum(cfd ** 2 for cfd in cfd_components)
        vec_rel_l2 = float(
            np.sqrt(np.mean(vec_diff_sq[velocity_valid]))
            / max(float(np.sqrt(np.mean(vec_ref_sq[velocity_valid]))), 1e-12)
        )
        velocity_metrics["vector_relative_l2"] = vec_rel_l2
        velocity_metrics["vector_similarity_score"] = float(np.clip(1.0 - vec_rel_l2, 0.0, 1.0))
    else:
        velocity_metrics = _field_similarity_metrics(np.zeros_like(mask, dtype=float), np.zeros_like(mask, dtype=float), velocity_valid)
        velocity_metrics["vector_relative_l2"] = float("nan")
        velocity_metrics["vector_similarity_score"] = float("nan")

    component_scores = [
        fields[name]["similarity_score"]
        for name in ["UR", "UT", "UZ", "P"]
        if np.isfinite(fields[name]["similarity_score"])
    ]
    if np.isfinite(velocity_metrics.get("vector_similarity_score", float("nan"))):
        component_scores.append(float(velocity_metrics["vector_similarity_score"]))
    overall = float(np.mean(component_scores)) if component_scores else float("nan")
    return {
        "description": "Full-grid CFD/NN similarity in fluid cells. similarity_score is clipped 1 - relative_l2.",
        "pressure_reference": pressure_reference,
        "field_metrics": fields,
        "velocity_metrics": velocity_metrics,
        "overall_similarity_score": overall,
    }

def compare_cfd_prediction_spans(
    self,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    spans: Sequence[float] = (0.2, 0.5, 0.8),
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    pressure_reference: str = "absolute",
    interpolation_chunk_size: int = 250_000,
) -> dict[str, Any]:
    # 对比图全部使用物理空间柱坐标 SI 单位。
    # 压力图回到原始 P；训练是否使用 P 值/梯度由 pressure_supervision_mode 单独控制。
    bundle = self._predict_case_bundle(case_index=case_index, case=case)
    case_data = bundle["case"]
    sample = bundle["sample"]
    pred_phy = bundle["pred_phy"]
    n = int(sample["x"].shape[-1])
    mask = sample["blade_mask"].detach().cpu().numpy()
    pred_p_for_plot = np.asarray(pred_phy["P"], dtype=np.float32)
    if pressure_reference == "training_origin":
        pred_p_for_plot = pred_p_for_plot - float(pred_p_for_plot[0, 0, 0])
    elif pressure_reference != "absolute":
        raise ValueError("pressure_reference must be 'training_origin' or 'absolute'.")

    field_names = ["UR", "UT", "UZ", "P"]
    p_label = "P [Pa]" if pressure_reference == "absolute" else "P rel. [Pa]"
    labels = {"UR": "UR [m/s]", "UT": "UT [m/s]", "UZ": "UZ [m/s]", "P": p_label}
    field_cmaps = {"UR": "coolwarm", "UT": "coolwarm", "UZ": "viridis", "P": "plasma"}
    total_rows = len(spans) * len(field_names)
    fig, axes = plt.subplots(total_rows, 3, figsize=(14, 3.0 * total_rows), squeeze=False)
    diagnostics: dict[str, Any] = {
        "pressure_supervision_mode": self.pressure_supervision_mode,
        "pressure_reference": pressure_reference,
        "units": {"UR": "m/s", "UT": "m/s", "UZ": "m/s", "P": "Pa"},
        "flow_similarity": _flow_similarity_diagnostics(
            pred_phy=pred_phy,
            case=case_data,
            mask=mask,
            pressure_reference=pressure_reference,
        ),
        "spans": {},
    }

    row = 0
    for span in spans:
        cfd_fields = _cfd_span_fields_from_case(
            case_data,
            span=float(span),
            n=n,
            interpolation_chunk_size=interpolation_chunk_size,
            pressure_reference=pressure_reference,
        )
        blade_mask = _span_slice_from_grid(mask, float(span)) > 0.5
        fluid = ~blade_mask
        span_key = f"{float(span):.6g}"
        diagnostics["spans"][span_key] = {}

        for name in field_names:
            if name == "P":
                pred_slice = _span_slice_from_grid(pred_p_for_plot, float(span))
                cfd_slice = cfd_fields["P"]
            else:
                pred_slice = _span_slice_from_grid(pred_phy[name], float(span))
                cfd_slice = cfd_fields[name]
            error = pred_slice - cfd_slice
            valid = fluid & np.isfinite(pred_slice) & np.isfinite(cfd_slice)
            if np.any(valid):
                err_valid = error[valid]
                cfd_valid = cfd_slice[valid]
                rmse = float(np.sqrt(np.mean(err_valid ** 2)))
                mae = float(np.mean(np.abs(err_valid)))
                denom = float(np.sqrt(np.mean(cfd_valid ** 2)) + 1e-12)
                rel_rmse = float(rmse / denom)
                max_abs = float(np.max(np.abs(err_valid)))
            else:
                rmse = mae = rel_rmse = max_abs = float("nan")
            diagnostics["spans"][span_key][name] = {
                "rmse": rmse,
                "mae": mae,
                "relative_rmse": rel_rmse,
                "max_abs_error": max_abs,
            }

            plot_mask = blade_mask.T
            cfd_plot = np.ma.array(cfd_slice.T, mask=plot_mask)
            pred_plot = np.ma.array(pred_slice.T, mask=plot_mask)
            err_plot = np.ma.array(error.T, mask=plot_mask)

            combined = np.concatenate([np.asarray(cfd_plot.compressed()), np.asarray(pred_plot.compressed())])
            if combined.size and np.any(np.isfinite(combined)):
                vmin = float(np.nanpercentile(combined, 2))
                vmax = float(np.nanpercentile(combined, 98))
                if abs(vmax - vmin) < 1e-12:
                    vmin, vmax = None, None
            else:
                vmin, vmax = None, None
            err_abs = float(np.nanpercentile(np.abs(err_plot.compressed()), 98)) if err_plot.count() else 1.0
            err_abs = max(err_abs, 1e-12)

            image0 = axes[row, 0].imshow(
                cfd_plot,
                origin="lower",
                aspect="auto",
                cmap=field_cmaps[name],
                vmin=vmin,
                vmax=vmax,
            )
            image1 = axes[row, 1].imshow(
                pred_plot,
                origin="lower",
                aspect="auto",
                cmap=field_cmaps[name],
                vmin=vmin,
                vmax=vmax,
            )
            image2 = axes[row, 2].imshow(
                err_plot,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-err_abs,
                vmax=err_abs,
            )
            for col, title in enumerate(["CFD", "NN", "NN - CFD"]):
                axes[row, col].contour(plot_mask.astype(float), levels=[0.5], colors="k", linewidths=0.6)
                axes[row, col].set_title(f"{title} {labels[name]} @ span={span:.2f}")
                axes[row, col].set_xlabel("Theta index")
                axes[row, col].set_ylabel("Z index")
            fig.colorbar(image0, ax=axes[row, 0], fraction=0.046, pad=0.04)
            fig.colorbar(image1, ax=axes[row, 1], fraction=0.046, pad=0.04)
            fig.colorbar(image2, ax=axes[row, 2], fraction=0.046, pad=0.04)
            row += 1

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"CFD/NN span 对比误差图已保存到: {save_path}")
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"CFD/NN span 误差统计已保存到: {summary_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return diagnostics

def post_process_case(
    self,
    case_index: int = 0,
    case: Mapping[str, Any] | None = None,
    spans: Sequence[float] = (0.2, 0.5, 0.8),
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    plot_3d: bool = False,
    save_path_3d: str | Path | None = None,
    passages_to_plot_3d: int | None = None,
    show_3d: bool | None = None,
) -> dict[str, Any]:
    # 这个 post 既能看训练样本，也能看一个全新的部署样本。
    bundle = self._predict_case_bundle(case_index=case_index, case=case)
    case_data = bundle["case"]
    sample = bundle["sample"]
    pred_dim = bundle["pred_dim"]
    pred_phy = bundle["pred_phy"]
    batch = bundle["batch"]

    pred_batch = {key: value.unsqueeze(0).to(self.device) for key, value in pred_dim.items()}
    q_pred = float(self.physics_loss.outlet_flow_rate(pred_batch["UZ"], batch).detach().cpu().view(-1)[0].item())
    q_target = float(sample["qv_passage"].item())
    train_n = int(self.train_dataset[0]["x"].shape[-1])
    deploy_n = int(sample["x"].shape[-1])
    ibm_c_profile = sample["ibm_C"].detach().cpu().reshape(-1)
    ibm_epsilon_profile = sample["ibm_epsilon"].detach().cpu().reshape(-1)
    ibm_c_mean, ibm_c_min, ibm_c_max = _profile_mean_min_max(ibm_c_profile)
    ibm_epsilon_mean, ibm_epsilon_min, ibm_epsilon_max = _profile_mean_min_max(ibm_epsilon_profile)

    print("\n========== 训练后 Post 检查 ==========")
    print(f"grid transfer: train_n={train_n}, deploy_n={deploy_n}")
    if deploy_n > train_n:
        print("当前展示的是“粗网格训练 -> 更细网格部署”的直接推理结果。")
    print(f"出口单流道体积流量: pred={q_pred:.6g}, target={q_target:.6g}")
    print(
        f"adaptive_ibm_profile : C mean/min/max={ibm_c_mean:.6g}/{ibm_c_min:.6g}/{ibm_c_max:.6g}, "
        f"epsilon mean/min/max={ibm_epsilon_mean:.6g}/{ibm_epsilon_min:.6g}/{ibm_epsilon_max:.6g}"
    )
    print("下面给出各个 span 处流体区域的 mean / min / max，单位已经回到物理空间(SI)。")
    mask = sample["blade_mask"] < 0.5
    n = mask.shape[0]
    for span in spans:
        r_index = span_to_index(span, n)
        fluid = mask[r_index]
        ur_stats = field_stats(pred_phy["UR"][r_index], fluid)
        ut_stats = field_stats(pred_phy["UT"][r_index], fluid)
        uz_stats = field_stats(pred_phy["UZ"][r_index], fluid)
        p_stats = field_stats(pred_phy["P"][r_index], fluid)
        c_at_span = float(ibm_c_profile[r_index].item())
        epsilon_at_span = float(ibm_epsilon_profile[r_index].item())

        print(f"span={span:.2f} (i={r_index}) | C={c_at_span:.6g}, epsilon={epsilon_at_span:.6g}")
        print(f"  UR [mean/min/max] = {ur_stats[0]:.6g} / {ur_stats[1]:.6g} / {ur_stats[2]:.6g}")
        print(f"  UT [mean/min/max] = {ut_stats[0]:.6g} / {ut_stats[1]:.6g} / {ut_stats[2]:.6g}")
        print(f"  UZ [mean/min/max] = {uz_stats[0]:.6g} / {uz_stats[1]:.6g} / {uz_stats[2]:.6g}")
        print(f"  P  [mean/min/max] = {p_stats[0]:.6g} / {p_stats[1]:.6g} / {p_stats[2]:.6g}")
    print("=====================================\n")

    fig, axes = plt.subplots(len(spans), 4, figsize=(18, 3.6 * len(spans)), squeeze=False)
    field_names = ["UR", "UT", "UZ", "P"]
    cmaps = {"UR": "coolwarm", "UT": "coolwarm", "UZ": "viridis", "P": "plasma"}

    for row, span in enumerate(spans):
        r_index = span_to_index(span, n)
        blade_mask = sample["blade_mask"][r_index].detach().cpu().numpy().T > 0.5
        for col, name in enumerate(field_names):
            data = pred_phy[name][r_index].detach().cpu().numpy().T
            data = np.ma.array(data, mask=blade_mask)
            image = axes[row, col].imshow(data, origin="lower", aspect="auto", cmap=cmaps[name])
            axes[row, col].contour(blade_mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
            axes[row, col].set_title(f"{name} @ span={span:.2f} (physical)")
            axes[row, col].set_xlabel("Theta index")
            axes[row, col].set_ylabel("Z index")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=160, bbox_inches="tight")
        print(f"后处理图已保存到: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    three_d_info = None
    if plot_3d:
        three_d_info = self.plot_3d_streamlines(
            case_index=case_index,
            case=case_data,
            show=show if show_3d is None else bool(show_3d),
            save_path=save_path_3d,
            passages_to_plot=passages_to_plot_3d,
        )

    return {
        "case": case_data,
        "sample": sample,
        "pred_dim": pred_dim,
        "pred_phy": pred_phy,
        "q_pred": q_pred,
        "q_target": q_target,
        "ibm_C": ibm_c_profile,
        "ibm_epsilon": ibm_epsilon_profile,
        "ibm_C_mean": ibm_c_mean,
        "ibm_epsilon_mean": ibm_epsilon_mean,
        "train_n": train_n,
        "deploy_n": deploy_n,
        "three_d_info": three_d_info,
    }
