from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_forward
from torch.utils.data import DataLoader, Dataset

import NeuralOperators
from BladeImport import build_blade_boundary
from SurrogateModelingConfig import FlowCaseConfig
from SurrogateModelingData import (
    find_unique_simulation_csv,
    make_pure_physics_debug_case,
    make_pure_physics_debug_cases,
    make_supervised_simulation_case,
    make_supervised_simulation_cases,
)
from SurrogateModelingKKT import apply_kkt_projection, build_kkt_projection
from SurrogateModelingPlots import (
    _axis_spectrum_energy as _axis_spectrum_energy_impl,
    _dct_along as _dct_along_impl,
    _trace_streamline_cylindrical as _trace_streamline_cylindrical_impl,
    compare_cfd_prediction_spans as _compare_cfd_prediction_spans_impl,
    frequency_energy_diagnostics as _frequency_energy_diagnostics_impl,
    plot_3d_streamlines as _plot_3d_streamlines_impl,
    plot_blade_spans as _plot_blade_spans_impl,
    plot_cfd_spans as _plot_cfd_spans_impl,
    plot_frequency_energy_trends as _plot_frequency_energy_trends_impl,
    plot_local_physics_residual_spans as _plot_local_physics_residual_spans_impl,
    plot_training_history as _plot_training_history_impl,
    post_process_case as _post_process_case_impl,
)
from SurrogateModelingUtils import (
    _case_span_profile,
    _expand_ibm_parameter,
    _pick,
    _profile_mean_min_max,
    _to_tensor,
    build_geometry_tensors,
    build_phi_from_signed_distance,
    case_summary,
    d1_periodic_with_overlap,
    d2_periodic_with_overlap,
    expand_scalar,
    hard_project_theta_periodic,
    line_quadrature_weight,
    neighbor_minus,
    neighbor_plus,
    normalize_target_fields,
    seed_everything,
)

# 这个文件负责把“叶片场 -> 流场”的代理建模流程接起来。
# 目前网络主体仍然调用 NeuralOperators.py 里的标准 2D FNO。
# 因此这里采用“沿 R 方向逐层共享权重”的方式：
# 每个半径层是一个 Theta-Z 平面，所有半径层共用同一套 2D FNO 权重。
#
# 输入方面默认同时使用 blade_mask 和 phi：
# 1. blade_mask 提供清晰的固体拓扑；
# 2. phi 提供适合硬约束和物理损失加权的平滑浸没边界信息。

FLUENT_CONTINUITY_FIRST5_DEFAULT = (1.0, 6.6045e-1, 5.3537e-1, 4.3327e-1, 3.6179e-1)


def _float_sequence_from_any(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        tokens = value.replace(";", ",").split(",")
    else:
        try:
            tokens = list(value)
        except TypeError:
            tokens = [value]
    result: list[float] = []
    for token in tokens:
        if token is None:
            continue
        text = str(token).strip()
        if text:
            result.append(float(text))
    return tuple(result)


def _resolve_fluent_continuity_scale(
    continuity_scale: float | None = None,
    continuity_first5: Sequence[float] | str | None = None,
) -> float | None:
    if continuity_scale is not None:
        value = float(continuity_scale)
        return value if value > 0.0 else None
    values = _float_sequence_from_any(continuity_first5)
    if not values:
        return None
    value = max(abs(item) for item in values)
    return float(value) if value > 0.0 else None


CONVECTION_INTERPOLATION_ALIASES = {
    "central": "central",
    "centered": "central",
    "upwind": "upwind",
    "upwind1": "upwind",
    "first_order_upwind": "upwind",
    "1st_order_upwind": "upwind",
    "upwind2": "upwind2",
    "second_order_upwind": "upwind2",
    "2nd_order_upwind": "upwind2",
}


def _normalize_convection_interpolation(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in CONVECTION_INTERPOLATION_ALIASES:
        return CONVECTION_INTERPOLATION_ALIASES[key]
    valid = ", ".join(sorted(set(CONVECTION_INTERPOLATION_ALIASES.values())))
    raise ValueError(f"convection_interpolation must be one of: {valid}.")


class BladeFlowDataset(Dataset):
    # 每个样本至少需要：
    # n, rh, rs, h, mu, rho, omega, qv, n_blade
    #
    # 叶片信息可以三选一：
    # 1. blade_params
    # 2. blade_mask
    # 3. phi
    #
    # 监督标签可选：
    # UR, UT, UZ, P
    def __init__(
        self,
        cases: Sequence[Mapping[str, Any]],
        input_mode: str = "both",
        pressure_reference: str = "training_origin",
    ):
        self.cases = list(cases)
        self.input_mode = input_mode
        self.pressure_reference = str(pressure_reference)
        if self.pressure_reference not in {"training_origin", "absolute"}:
            raise ValueError("pressure_reference must be 'training_origin' or 'absolute'.")
        self.samples = [self._build_sample(case) for case in self.cases]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def _build_sample(self, case: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        config = FlowCaseConfig.from_mapping(case)
        geometry = build_geometry_tensors(config)
        ibm_c_profile = _case_span_profile(
            case,
            ("ibm_C_profile", "ibm_C_span", "ibm_C_r", "ibm_C"),
            n=config.n,
            default=config.ibm_C,
        )
        ibm_epsilon_profile = _case_span_profile(
            case,
            ("ibm_epsilon_profile", "ibm_epsilon_span", "ibm_epsilon_r", "ibm_epsilon"),
            n=config.n,
            default=config.ibm_epsilon,
        )
        blade_mask, phi, signed_distance = self._build_blade_channels(
            case,
            config,
            ibm_c_profile=ibm_c_profile,
            ibm_epsilon_profile=ibm_epsilon_profile,
        )
        target, has_target = normalize_target_fields(case, config, pressure_reference=self.pressure_reference)
        has_true_signed_distance = bool(_pick(case, "signed_distance") is not None or _pick(case, "blade_params") is not None)

        # 输入通道分两部分：
        # 一部分是叶片场本身；另一部分是几何系数和无量纲常数场。
        channels: list[torch.Tensor] = []
        if self.input_mode == "mask":
            channels.append(blade_mask)
        elif self.input_mode == "phi":
            channels.append(phi)
        elif self.input_mode == "both":
            channels.append(blade_mask)
            channels.append(phi)
        else:
            raise ValueError("input_mode must be 'mask', 'phi', or 'both'")

        # 把几何因素和工况参数也塞进去
        channels.extend(
            [
                geometry["r_hat"],
                geometry["K_theta"],
                geometry["Theta_sin"],
                geometry["Theta_cos"],
                geometry["Z"],
                geometry["solid_ut"],
                torch.full_like(geometry["r_hat"], config.Eu_omega),
                torch.full_like(geometry["r_hat"], config.Re_omega),
                torch.full_like(geometry["r_hat"], config.Lambda),
                torch.full_like(geometry["r_hat"], config.Ku),
                torch.full_like(geometry["r_hat"], config.delta),
                torch.full_like(geometry["r_hat"], config.sgn_omega),
                torch.full_like(geometry["r_hat"], config.g_star),
            ]
        )
        x = torch.stack(channels, dim=0).to(torch.float32)

        # 输入在进入网络前就先做一次拼缝投影，避免 seam 两侧自相矛盾。
        x = hard_project_theta_periodic(x, theta_dim=2)

        return {
            "x": x,
            "y": target,
            "has_target": has_target,
            "phi": phi.to(torch.float32),
            "blade_mask": blade_mask.to(torch.float32),
            "signed_distance": signed_distance.to(torch.float32),
            "has_true_signed_distance": torch.tensor(1.0 if has_true_signed_distance else 0.0, dtype=torch.float32),
            "solid_ut": geometry["solid_ut"].to(torch.float32),
            "r_hat": geometry["r_hat"].to(torch.float32),
            "K_theta": geometry["K_theta"].to(torch.float32),
            "R": geometry["R"].to(torch.float32),
            "Theta": geometry["Theta"].to(torch.float32),
            "Z": geometry["Z"].to(torch.float32),
            "dR": torch.tensor(config.dR, dtype=torch.float32),
            "dTheta": torch.tensor(config.dTheta, dtype=torch.float32),
            "dZ": torch.tensor(config.dZ, dtype=torch.float32),
            "Eu_omega": torch.tensor(config.Eu_omega, dtype=torch.float32),
            "Re_omega": torch.tensor(config.Re_omega, dtype=torch.float32),
            "Lambda": torch.tensor(config.Lambda, dtype=torch.float32),
            "Ku": torch.tensor(config.Ku, dtype=torch.float32),
            "delta": torch.tensor(config.delta, dtype=torch.float32),
            "sgn_omega": torch.tensor(config.sgn_omega, dtype=torch.float32),
            "theta0": torch.tensor(config.theta0, dtype=torch.float32),
            "delta_r": torch.tensor(config.delta_r, dtype=torch.float32),
            "u_omega": torch.tensor(config.u_omega, dtype=torch.float32),
            "u_zo": torch.tensor(config.u_zo, dtype=torch.float32),
            "P0": torch.tensor(config.P0, dtype=torch.float32),
            "qv": torch.tensor(config.qv, dtype=torch.float32),
            "n_blade": torch.tensor(float(config.n_blade), dtype=torch.float32),
            "qv_passage": torch.tensor(config.qv_passage, dtype=torch.float32),
            "qv_hat": torch.tensor(config.qv_hat, dtype=torch.float32),
            "g_star": torch.tensor(config.g_star, dtype=torch.float32),
            "ibm_C": ibm_c_profile.to(torch.float32),
            "ibm_epsilon": ibm_epsilon_profile.to(torch.float32),
            "absolute_frame": torch.tensor(1.0 if config.absolute_frame else 0.0, dtype=torch.float32),
        }

    def _build_blade_channels(
        self,
        case: Mapping[str, Any],
        config: FlowCaseConfig,
        *,
        ibm_c_profile: torch.Tensor,
        ibm_epsilon_profile: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_in = _pick(case, "phi")
        mask_in = _pick(case, "blade_mask", "mask")
        signed_distance_in = _pick(case, "signed_distance")

        phi = _to_tensor(phi_in, dtype=torch.float32) if phi_in is not None else None
        blade_mask = _to_tensor(mask_in, dtype=torch.float32) if mask_in is not None else None
        signed_distance = _to_tensor(signed_distance_in, dtype=torch.float32) if signed_distance_in is not None else None

        # 如果没有直接给 mask / phi，就尝试从 blade_params 现场重建。
        if blade_mask is None or phi is None:
            blade_params = _pick(case, "blade_params")
            if blade_params is not None:
                boundary = build_blade_boundary(blade_params, config.make_passage_geometry())
                blade_mask = _to_tensor(boundary.mask.astype(np.float32))
                signed_distance = _to_tensor(boundary.signed_distance)

        if blade_mask is None and phi is None:
            raise ValueError("Each case must provide blade_params, blade_mask, or phi.")

        if blade_mask is None and phi is not None:
            blade_mask = (phi < 0.5).to(torch.float32)

        if phi is None and signed_distance is not None:
            phi = build_phi_from_signed_distance(signed_distance, ibm_c_profile, ibm_epsilon_profile)

        if phi is None:
            # 如果只有 sharp mask，就退化成最简单的 0/1 phi。
            phi = 1.0 - blade_mask

        if signed_distance is None:
            # 没有真正的距离函数时，先给一个粗略的占位量，方便后续调试。
            signed_distance = 1.0 - 2.0 * blade_mask

        blade_mask = (blade_mask > 0.5).to(torch.float32)
        phi = torch.clamp(phi.to(torch.float32), min=0.0, max=1.0)
        signed_distance = signed_distance.to(torch.float32)

        # 叶片场本身也要满足周向拼缝一致，否则后面周期差分一定会在 seam 上出假残差。
        blade_mask = hard_project_theta_periodic(blade_mask, theta_dim=1)
        phi = hard_project_theta_periodic(phi, theta_dim=1)
        signed_distance = hard_project_theta_periodic(signed_distance, theta_dim=1)

        blade_mask = (blade_mask > 0.5).to(torch.float32)
        return blade_mask, phi, signed_distance


class SliceWiseFNOFlowModel(nn.Module):
    # 这里不直接写 3D FNO，而是沿 R 方向逐层调用 2D FNO。
    #
    # 还有一个关键细节：
    # NeuralOperators.py 里的 Fourier 卷积在两个方向上都天然带“周期卷积”味道，
    # 但本问题只有 Theta 是周期，Z 并不是周期。
    # 所以这里在 Z 方向做复制填充，再把中心区域裁回来，尽量减弱入口/出口的人为 wrap-around。
    def __init__(
        self,
        input_channels: int,
        modes: int = 8,
        width: int = 16,
        depth: int = 4,
        z_padding: int = 8,
        operator_variant: str = "hf_cfno",
        high_modes: int | None = None,
        fourier_feature_bands: Sequence[int] = (1, 2, 4, 8),
        hf_high_gate_init: float = -1.0,
        hf_use_local_highpass: bool = True,
        pressure_smoothing: float = 0.0,
        pressure_reference_mode: str = "origin",
        slice_batch_size: int | None = None,
        auto_cuda_batching: bool = True,
        cuda_memory_fraction: float = 0.65,
        use_activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.z_padding = int(max(z_padding, 0))
        self.pressure_smoothing = float(np.clip(pressure_smoothing, 0.0, 1.0))
        self.pressure_reference_mode = str(pressure_reference_mode).lower()
        if self.pressure_reference_mode not in {"origin", "none"}:
            raise ValueError("pressure_reference_mode must be 'origin' or 'none'.")
        self.operator_variant = str(operator_variant).lower()
        self.is_3d_operator = False
        self.width = int(width)
        self.depth = int(depth)
        self.slice_batch_size = None if slice_batch_size is None else max(1, int(slice_batch_size))
        self.auto_cuda_batching = bool(auto_cuda_batching)
        self.cuda_memory_fraction = float(np.clip(cuda_memory_fraction, 0.05, 0.95))
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self._runtime_slice_batch_size: int | None = self.slice_batch_size
        # todo 切换模型看这个
        if self.operator_variant in {"fno", "legacy_fno"}:
            self.core = NeuralOperators.FNO2d_small(
                modes=modes,
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"cno", "legacy_cno"}:
            self.core = NeuralOperators.CNO2d_small(
                cheb_modes=(modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"wno", "wno2d", "wavelet", "wavelet2d"}:
            self.core = NeuralOperators.WNO2d_small(
                modes=modes,
                wavelet_modes=(modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"cfno", "legacy_cfno"}:
            self.core = NeuralOperators.CFNO2d_small(
                modes=modes,
                cheb_modes=(modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"hf_cfno", "high_frequency_cfno"}:
            self.core = NeuralOperators.HF_CFNO2d_small(
                modes=modes,
                cheb_modes=(modes, modes),
                high_modes=high_modes,
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
                fourier_feature_bands=tuple(int(v) for v in fourier_feature_bands),
                high_gate_init=hf_high_gate_init,
                use_local_highpass=hf_use_local_highpass,
            )
        elif self.operator_variant in {"fno3d", "fno_3d", "3d_fno"}:
            self.is_3d_operator = True
            self.core = NeuralOperators.FNO3d_small(
                modes=modes,
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"cno3d", "cno_3d", "3d_cno"}:
            self.is_3d_operator = True
            self.core = NeuralOperators.CNO3d_small(
                cheb_modes=(modes, modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"wno3d", "wno_3d", "3d_wno", "wavelet3d", "wavelet_3d"}:
            self.is_3d_operator = True
            self.core = NeuralOperators.WNO3d_small(
                modes=modes,
                wavelet_modes=(modes, modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"cfno3d", "cfno_3d", "3d_cfno"}:
            self.is_3d_operator = True
            self.core = NeuralOperators.CFNO3d_small(
                modes=modes,
                cheb_modes=(modes, modes, modes),
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
            )
        elif self.operator_variant in {"hf_cfno3d", "hf_cfno_3d", "high_frequency_cfno3d", "3d_hf_cfno"}:
            self.is_3d_operator = True
            self.core = NeuralOperators.HF_CFNO3d_small(
                modes=modes,
                cheb_modes=(modes, modes, modes),
                high_modes=high_modes,
                width=width,
                depth=depth,
                input_features=input_channels,
                output_features=4,
                fourier_feature_bands=tuple(int(v) for v in fourier_feature_bands),
                high_gate_init=hf_high_gate_init,
                use_local_highpass=hf_use_local_highpass,
            )
        else:
            raise ValueError(
                "operator_variant must be one of: "
                "'fno', 'cno', 'wno', 'cfno', 'hf_cfno', "
                "'fno3d', 'cno3d', 'wno3d', 'cfno3d', 'hf_cfno3d'."
            )

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return (
            "cuda out of memory" in message
            or "cublas_status_alloc_failed" in message
            or "cudnn_status_alloc_failed" in message
        )

    @staticmethod
    def _empty_cuda_cache_for(x: torch.Tensor) -> None:
        if x.is_cuda:
            torch.cuda.empty_cache()

    def _call_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint_forward(self.core, x, use_reentrant=False)
        return self.core(x)

    def _estimate_slice_batch_size(self, x_slice: torch.Tensor) -> int:
        total_slices, in_channels, n_theta, n_z = x_slice.shape
        if total_slices <= 1:
            return total_slices
        if self._runtime_slice_batch_size is not None:
            return min(total_slices, self._runtime_slice_batch_size)
        if not (self.auto_cuda_batching and x_slice.is_cuda):
            return total_slices

        free_bytes, _ = torch.cuda.mem_get_info(x_slice.device)
        budget = max(1, int(free_bytes * self.cuda_memory_fraction))
        element_size = max(int(x_slice.element_size()), 1)
        width = max(self.width, in_channels, 4)
        activation_factor = max(18, 8 * self.depth)
        bytes_per_slice = n_theta * n_z * element_size * (in_channels + 4 + activation_factor * width)
        estimate = max(1, int(budget // max(bytes_per_slice, 1)))
        return min(total_slices, estimate)

    def _run_core_2d_batched(self, x_slice: torch.Tensor) -> torch.Tensor:
        total_slices = int(x_slice.shape[0])
        slice_batch = max(1, self._estimate_slice_batch_size(x_slice))
        if slice_batch >= total_slices:
            return self._call_core(x_slice)

        outputs: list[torch.Tensor] = []
        start = 0
        while start < total_slices:
            current = min(slice_batch, total_slices - start)
            while True:
                try:
                    outputs.append(self._call_core(x_slice[start:start + current]))
                    break
                except RuntimeError as exc:
                    if not (self.auto_cuda_batching and self._is_cuda_oom(exc) and current > 1):
                        raise
                    current = max(1, current // 2)
                    self._runtime_slice_batch_size = current
                    slice_batch = current
                    self._empty_cuda_cache_for(x_slice)
                    print(f"CUDA 显存不足：2D operator slice batch 自动降到 {current}。")
            start += current
        return torch.cat(outputs, dim=0)

    def smooth_pressure(self, p: torch.Tensor) -> torch.Tensor:
        r_plus = neighbor_plus(p, dim=1, periodic=False)
        r_minus = neighbor_minus(p, dim=1, periodic=False)
        theta_plus = neighbor_plus(p, dim=2, periodic=True)
        theta_minus = neighbor_minus(p, dim=2, periodic=True)
        z_plus = neighbor_plus(p, dim=3, periodic=False)
        z_minus = neighbor_minus(p, dim=3, periodic=False)
        return (p + r_plus + r_minus + theta_plus + theta_minus + z_plus + z_minus) / 7.0

    def forward(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
        solid_ut: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # x shape = [B, C, R, Theta, Z]
        batch_size, in_channels, n_r, n_theta, n_z = x.shape

        # 先把输入拼缝再对齐一次，确保进入 FNO 的每个 Theta-Z 切片是周期一致的。
        x = hard_project_theta_periodic(x, theta_dim=3)

        # 每个半径层对应一个 Theta-Z 平面。
        if self.is_3d_operator:
            x_core = x
            if self.z_padding > 0:
                x_core = F.pad(x_core, (self.z_padding, self.z_padding, 0, 0, 0, 0), mode="replicate")

            raw = self._call_core(x_core)

            if self.z_padding > 0:
                raw = raw[..., self.z_padding:-self.z_padding]
        else:
            x_slice = x.permute(0, 2, 1, 3, 4).reshape(batch_size * n_r, in_channels, n_theta, n_z)

            if self.z_padding > 0:
                x_slice = F.pad(x_slice, (self.z_padding, self.z_padding, 0, 0), mode="replicate")

            raw = self._run_core_2d_batched(x_slice)

            if self.z_padding > 0:
                raw = raw[..., self.z_padding:-self.z_padding]

            raw = raw.reshape(batch_size, n_r, 4, n_theta, n_z).permute(0, 2, 1, 3, 4)

        if raw.shape != (batch_size, 4, n_r, n_theta, n_z):
            raise RuntimeError(
                "Neural operator returned an unexpected shape: "
                f"expected {(batch_size, 4, n_r, n_theta, n_z)}, got {tuple(raw.shape)}."
            )

        ur_raw = raw[:, 0]
        ut_raw = raw[:, 1]
        uz_raw = raw[:, 2]
        p_raw = raw[:, 3]

        # 用 phi 做硬约束：
        # 1. 叶片内部 UR=0, UZ=0；
        # 2. 叶片内部 UT=U_solid，表示绝对参考系中的旋转无滑移。
        solid = 1.0 - phi
        ur = phi * ur_raw
        ut = phi * ut_raw + solid * solid_ut
        uz = phi * uz_raw
        p = hard_project_theta_periodic(p_raw, theta_dim=2)
        if self.pressure_smoothing > 0.0:
            p = (1.0 - self.pressure_smoothing) * p + self.pressure_smoothing * self.smooth_pressure(p)
            p = hard_project_theta_periodic(p, theta_dim=2)

        # hub / shroud 也做硬约束。
        ur[:, 0, :, :] = 0.0
        ur[:, -1, :, :] = 0.0
        uz[:, 0, :, :] = 0.0
        uz[:, -1, :, :] = 0.0
        ut[:, 0, :, :] = solid_ut[:, 0, :, :]
        ut[:, -1, :, :] = solid_ut[:, -1, :, :]

        # 所有输出场都强制首尾拼缝一致。
        ur = hard_project_theta_periodic(ur, theta_dim=2)
        ut = hard_project_theta_periodic(ut, theta_dim=2)
        uz = hard_project_theta_periodic(uz, theta_dim=2)
        p = hard_project_theta_periodic(p, theta_dim=2)

        # 压力参考点可选固定在 (0,0,0)。
        # mixed 训练若直接用 CFD 的 P 值监督，可关闭这个 gauge fixing，让数据决定压力零点。
        if self.pressure_reference_mode == "origin":
            p_ref = p[:, 0, 0, 0].view(batch_size, 1, 1, 1)
            p = p - p_ref

        return {
            "UR": ur,
            "UT": ut,
            "UZ": uz,
            "P": p,
        }


class AdaptiveIBMMaskController(nn.Module):
    # Current version learns span-wise profiles C(r) and epsilon(r).
    # 这里不把 ibm_C 和 ibm_epsilon 当成完全写死的常数。
    # 做法是：给它们一个允许范围，然后根据当前样本的几何/工况特征，
    # 学出每个样本各自的一对“等效 IBM 过渡层参数”。
    #
    # 注意这里仍然保持“单个样本内空间上统一”的 C / epsilon。
    # 也就是说，我们先从“不同流场、不同叶型可以不同”这一步做起，
    # 不直接把它扩成空间分布场，避免把训练问题一下子搞得过硬。
    def __init__(
        self,
        *,
        c_range: tuple[float, float] = (0.25, 4.0),
        epsilon_range: tuple[float, float] = (0.01, 0.08),
        hidden: int = 16,
        default_c: float = 1.0,
        default_epsilon: float = 0.025,
    ):
        super().__init__()

        c_min, c_max = float(c_range[0]), float(c_range[1])
        epsilon_min, epsilon_max = float(epsilon_range[0]), float(epsilon_range[1])
        if not (c_max > c_min and epsilon_max > epsilon_min):
            raise ValueError("ibm parameter ranges must satisfy max > min.")

        self.c_range = (c_min, c_max)
        self.epsilon_range = (epsilon_min, epsilon_max)
        self.hidden = int(hidden)

        default_c_ratio = np.clip((float(default_c) - c_min) / (c_max - c_min), 1e-4, 1.0 - 1e-4)
        default_epsilon_ratio = np.clip(
            (float(default_epsilon) - epsilon_min) / (epsilon_max - epsilon_min),
            1e-4,
            1.0 - 1e-4,
        )
        base_logit = torch.tensor(
            [
                np.log(default_c_ratio / (1.0 - default_c_ratio)),
                np.log(default_epsilon_ratio / (1.0 - default_epsilon_ratio)),
            ],
            dtype=torch.float32,
        )
        self.base_logit = nn.Parameter(base_logit)

        self.mlp = nn.Sequential(
            nn.Linear(9, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 2),
        )

        # 让训练从 case 默认值附近起步，而不是一开始就把 phi 扭得很厉害。
        last_layer = self.mlp[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _map_to_range(self, raw_value: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(raw_value)

    def extract_features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # 用少量全局特征描述这个样本的“几何 + 流动尺度”。
        blade_fraction = torch.mean(batch["blade_mask"], dim=(2, 3))
        signed_distance_abs = torch.abs(batch["signed_distance"])
        signed_distance_mean = torch.mean(signed_distance_abs, dim=(2, 3))
        signed_distance_std = torch.std(signed_distance_abs, dim=(2, 3), unbiased=False)
        span_r = batch["R"][:, :, 0, 0]

        qv_hat = batch["qv_hat"].view(-1, 1).expand_as(span_r)
        re_log = torch.log(torch.clamp(batch["Re_omega"].view(-1, 1), min=1e-12)).expand_as(span_r)
        eu_log = torch.log(torch.clamp(batch["Eu_omega"].view(-1, 1), min=1e-12)).expand_as(span_r)
        lambda_value = batch["Lambda"].view(-1, 1).expand_as(span_r)
        delta_value = batch["delta"].view(-1, 1).expand_as(span_r)

        return torch.stack(
            [
                span_r,
                blade_fraction,
                signed_distance_mean,
                signed_distance_std,
                qv_hat,
                re_log,
                eu_log,
                lambda_value,
                delta_value,
            ],
            dim=-1,
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(batch)
        offset = 0.5 * torch.tanh(self.mlp(features))
        raw_value = self.base_logit.view(1, 1, 2) + offset

        ibm_c = self._map_to_range(raw_value[..., 0], self.c_range[0], self.c_range[1]).unsqueeze(-1).unsqueeze(-1)
        ibm_epsilon = self._map_to_range(
            raw_value[..., 1],
            self.epsilon_range[0],
            self.epsilon_range[1],
        ).unsqueeze(-1).unsqueeze(-1)
        return ibm_c, ibm_epsilon


class BladeFlowPhysicsLoss(nn.Module):
    # 物理损失由五部分组成：
    # 1. 连续方程残差
    # 2. 径向动量方程残差
    # 3. 周向动量方程残差
    # 4. 轴向动量方程残差
    # 5. 出口单流道体积流量约束
    #
    # 此外还额外输出两个诊断量：
    # - 周向周期拼缝误差
    # - 叶片无滑移误差
    #
    # 这两个诊断量默认不计入总损失，因为前向里已经做了硬约束。
    def __init__(
        self,
        pressure_highpass_weight: float = 0.0,
        pressure_highpass_normalized: bool = True,
        physics_discretization: str = "fvm_rhie_chow",
        rhie_chow_strength: float = 0.35,
        momentum_diagonal_floor: float = 1.0,
        convection_interpolation: str = "central",
        fluent_continuity_first5: Sequence[float] | str | None = None,
        fluent_continuity_scale: float | None = None,
    ):
        super().__init__()
        self.pressure_highpass_weight = float(max(pressure_highpass_weight, 0.0))
        self.pressure_highpass_normalized = bool(pressure_highpass_normalized)
        self.physics_discretization = str(physics_discretization).lower()
        if self.physics_discretization not in {"fvm_rhie_chow", "centered"}:
            raise ValueError("physics_discretization must be 'fvm_rhie_chow' or 'centered'.")
        self.rhie_chow_strength = float(max(rhie_chow_strength, 0.0))
        self.momentum_diagonal_floor = float(max(momentum_diagonal_floor, 1e-8))
        self.convection_interpolation = _normalize_convection_interpolation(convection_interpolation)
        self.fluent_continuity_first5 = _float_sequence_from_any(fluent_continuity_first5)
        self.fluent_continuity_scale = _resolve_fluent_continuity_scale(
            fluent_continuity_scale,
            self.fluent_continuity_first5,
        )

    def d1(
        self,
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
        periodic: bool,
        duplicate_endpoint: bool = False,
    ) -> torch.Tensor:
        if periodic and duplicate_endpoint:
            return d1_periodic_with_overlap(x, dim, spacing)
        xp = neighbor_plus(x, dim, periodic)
        xm = neighbor_minus(x, dim, periodic)
        return (xp - xm) / (2.0 * spacing)

    def d2(
        self,
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
        periodic: bool,
        duplicate_endpoint: bool = False,
    ) -> torch.Tensor:
        if periodic and duplicate_endpoint:
            return d2_periodic_with_overlap(x, dim, spacing)
        xp = neighbor_plus(x, dim, periodic)
        xm = neighbor_minus(x, dim, periodic)
        return (xp - 2.0 * x + xm) / (spacing ** 2)

    def weighted_mse(self, residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.sum((residual ** 2) * weight) / torch.clamp(torch.sum(weight), min=1e-12)

    def scaled_l1_residual(
        self,
        residual: torch.Tensor,
        equation_scale: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        numerator = torch.sum(torch.abs(residual) * weight)
        denominator = torch.sum(torch.abs(equation_scale) * weight)
        return numerator / torch.clamp(denominator, min=1e-30)

    def _centered_face_velocities(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "ur_e": self._face_plus(ur, dim=1),
            "ur_w": self._face_minus(ur, dim=1),
            "ut_n": self._face_plus(ut, dim=2, theta_overlap=True),
            "ut_s": self._face_minus(ut, dim=2, theta_overlap=True),
            "uz_t": self._face_plus(uz, dim=3),
            "uz_b": self._face_minus(uz, dim=3),
        }

    def _continuity_equation_scale(
        self,
        face_velocity: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.fluent_continuity_scale is not None:
            # Fluent continuity is scaled by a reference continuity imbalance
            # taken from the first iterations. The monitor values users usually
            # paste are already scaled; in that case max(first5)=1 simply makes
            # this reported metric the mean absolute residual in our units.
            return torch.full_like(batch["r_hat"], float(self.fluent_continuity_scale))
        # NN training has no SIMPLE first-five-iteration baseline, so continuity uses a current mass-flux scale.
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        r_e = self._face_plus(r_hat, dim=1)
        r_w = self._face_minus(r_hat, dim=1)

        radial = (torch.abs(r_e * face_velocity["ur_e"]) + torch.abs(r_w * face_velocity["ur_w"])) / (
            torch.clamp(r_hat, min=1e-12) * torch.clamp(dR, min=1e-12)
        )
        theta = torch.abs(k_theta) * (torch.abs(face_velocity["ut_n"]) + torch.abs(face_velocity["ut_s"])) / torch.clamp(
            dTheta,
            min=1e-12,
        )
        axial = torch.abs(Lambda * Ku) * (
            torch.abs(face_velocity["uz_t"]) + torch.abs(face_velocity["uz_b"])
        ) / torch.clamp(dZ, min=1e-12)
        return radial + theta + axial

    def _momentum_diagonal_scale(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # a_P scale used only for reporting Fluent-like residuals; training still uses raw MSE losses.
        a_inv = self._momentum_diagonal_inverse(ur, ut, uz, batch)
        return torch.reciprocal(torch.clamp(a_inv, min=1e-30))

    def _momentum_velocity_magnitude_scales(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Fluent's pressure-based global scaling replaces a_P * phi_P with
        # a_P * |V| for momentum equations. Our variables are nondimensionalized
        # with u_omega for UR/UT and u_zo for UZ, so the same physical |V| maps
        # to different dimensionless scales for radial/theta and axial residuals.
        Ku = torch.clamp(torch.abs(expand_scalar(batch["Ku"])), min=1e-12)
        velocity_mag_omega = torch.sqrt(torch.clamp(ur ** 2 + ut ** 2 + (Ku * uz) ** 2, min=1e-30))
        velocity_mag_zo = torch.sqrt(torch.clamp((ur / Ku) ** 2 + (ut / Ku) ** 2 + uz ** 2, min=1e-30))
        return velocity_mag_omega.detach(), velocity_mag_zo.detach()

    def scaled_residual_metrics(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
        residual_c: torch.Tensor,
        residual_r: torch.Tensor,
        residual_theta: torch.Tensor,
        residual_z: torch.Tensor,
        pde_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ur = pred["UR"]
        ut = pred["UT"]
        uz = pred["UZ"]
        p = pred["P"]

        if self.physics_discretization == "fvm_rhie_chow":
            face_velocity = self._rhie_chow_face_velocities(ur, ut, uz, p, batch)
        else:
            face_velocity = self._centered_face_velocities(ur, ut, uz)

        continuity_scale = self._continuity_equation_scale(face_velocity, batch)
        momentum_diagonal = self._momentum_diagonal_scale(ur, ut, uz, batch)
        velocity_mag_omega, velocity_mag_zo = self._momentum_velocity_magnitude_scales(ur, ut, uz, batch)
        scaled_r = self.scaled_l1_residual(residual_r, momentum_diagonal * velocity_mag_omega, pde_mask)
        scaled_theta = self.scaled_l1_residual(residual_theta, momentum_diagonal * velocity_mag_omega, pde_mask)
        scaled_z = self.scaled_l1_residual(residual_z, momentum_diagonal * velocity_mag_zo, pde_mask)
        scaled_momentum = torch.mean(torch.stack([scaled_r, scaled_theta, scaled_z]))

        return {
            "scaled_res_c": self.scaled_l1_residual(residual_c, continuity_scale, pde_mask),
            "scaled_res_r": scaled_r,
            "scaled_res_theta": scaled_theta,
            "scaled_res_z": scaled_z,
            "scaled_res_momentum": scaled_momentum,
            "scaled_res_c_reference": torch.tensor(
                float(self.fluent_continuity_scale) if self.fluent_continuity_scale is not None else float("nan"),
                device=scaled_momentum.device,
                dtype=scaled_momentum.dtype,
            ),
        }

    def pressure_highpass(self, p: torch.Tensor) -> torch.Tensor:
        r_plus = neighbor_plus(p, dim=1, periodic=False)
        r_minus = neighbor_minus(p, dim=1, periodic=False)
        theta_plus = neighbor_plus(p, dim=2, periodic=True)
        theta_minus = neighbor_minus(p, dim=2, periodic=True)
        z_plus = neighbor_plus(p, dim=3, periodic=False)
        z_minus = neighbor_minus(p, dim=3, periodic=False)
        smooth = (p + r_plus + r_minus + theta_plus + theta_minus + z_plus + z_minus) / 7.0
        return p - smooth

    def build_pde_mask(self, phi: torch.Tensor) -> torch.Tensor:
        # 残差只在流体区内部统计：
        # 1. 叶片区由 phi 抑制；
        # 2. R 边界与 Z 边界单独由硬约束或弱约束处理，因此这里不计入；
        # 3. Theta 首尾是重合点，给半权避免拼缝位置被重复统计。
        mask = phi.clone()
        mask[:, 0, :, :] = 0.0
        mask[:, -1, :, :] = 0.0
        mask[:, :, :, 0] = 0.0
        mask[:, :, :, -1] = 0.0

        theta_weight = line_quadrature_weight(mask.shape[2], mask.device, mask.dtype)
        mask = mask * theta_weight.view(1, 1, -1, 1)
        return mask

    def _neighbor_plus_overlap(self, x: torch.Tensor, dim: int, *, theta_overlap: bool) -> torch.Tensor:
        if not theta_overlap:
            return neighbor_plus(x, dim=dim, periodic=False)
        x_perm = torch.movedim(x, dim, -1)
        out = torch.empty_like(x_perm)
        n = x_perm.shape[-1]
        if n <= 1:
            out.copy_(x_perm)
        elif n == 2:
            out[..., 0] = x_perm[..., 1]
            out[..., 1] = x_perm[..., 0]
        else:
            out[..., :-1] = x_perm[..., 1:]
            out[..., -1] = x_perm[..., 1]
        return torch.movedim(out, -1, dim)

    def _neighbor_minus_overlap(self, x: torch.Tensor, dim: int, *, theta_overlap: bool) -> torch.Tensor:
        if not theta_overlap:
            return neighbor_minus(x, dim=dim, periodic=False)
        x_perm = torch.movedim(x, dim, -1)
        out = torch.empty_like(x_perm)
        n = x_perm.shape[-1]
        if n <= 1:
            out.copy_(x_perm)
        elif n == 2:
            out[..., 0] = x_perm[..., 1]
            out[..., 1] = x_perm[..., 0]
        else:
            out[..., 1:] = x_perm[..., :-1]
            out[..., 0] = x_perm[..., -2]
        return torch.movedim(out, -1, dim)

    def _face_plus(self, x: torch.Tensor, dim: int, *, theta_overlap: bool = False) -> torch.Tensor:
        return 0.5 * (x + self._neighbor_plus_overlap(x, dim, theta_overlap=theta_overlap))

    def _face_minus(self, x: torch.Tensor, dim: int, *, theta_overlap: bool = False) -> torch.Tensor:
        return 0.5 * (self._neighbor_minus_overlap(x, dim, theta_overlap=theta_overlap) + x)

    def _upwind_plus(
        self,
        transported: torch.Tensor,
        normal_velocity: torch.Tensor,
        dim: int,
        *,
        theta_overlap: bool = False,
    ) -> torch.Tensor:
        plus = self._neighbor_plus_overlap(transported, dim, theta_overlap=theta_overlap)
        return torch.where(normal_velocity >= 0.0, transported, plus)

    def _upwind_minus(
        self,
        transported: torch.Tensor,
        normal_velocity: torch.Tensor,
        dim: int,
        *,
        theta_overlap: bool = False,
    ) -> torch.Tensor:
        minus = self._neighbor_minus_overlap(transported, dim, theta_overlap=theta_overlap)
        return torch.where(normal_velocity >= 0.0, minus, transported)

    def _upwind2_plus(
        self,
        transported: torch.Tensor,
        normal_velocity: torch.Tensor,
        dim: int,
        *,
        theta_overlap: bool = False,
    ) -> torch.Tensor:
        min_points = 4 if theta_overlap else 3
        if transported.shape[dim] < min_points:
            return self._upwind_plus(transported, normal_velocity, dim, theta_overlap=theta_overlap)
        minus = self._neighbor_minus_overlap(transported, dim, theta_overlap=theta_overlap)
        plus = self._neighbor_plus_overlap(transported, dim, theta_overlap=theta_overlap)
        plus2 = self._neighbor_plus_overlap(plus, dim, theta_overlap=theta_overlap)
        from_left = 1.5 * transported - 0.5 * minus
        from_right = 1.5 * plus - 0.5 * plus2
        return torch.where(normal_velocity >= 0.0, from_left, from_right)

    def _upwind2_minus(
        self,
        transported: torch.Tensor,
        normal_velocity: torch.Tensor,
        dim: int,
        *,
        theta_overlap: bool = False,
    ) -> torch.Tensor:
        min_points = 4 if theta_overlap else 3
        if transported.shape[dim] < min_points:
            return self._upwind_minus(transported, normal_velocity, dim, theta_overlap=theta_overlap)
        minus = self._neighbor_minus_overlap(transported, dim, theta_overlap=theta_overlap)
        minus2 = self._neighbor_minus_overlap(minus, dim, theta_overlap=theta_overlap)
        plus = self._neighbor_plus_overlap(transported, dim, theta_overlap=theta_overlap)
        from_left = 1.5 * minus - 0.5 * minus2
        from_right = 1.5 * transported - 0.5 * plus
        return torch.where(normal_velocity >= 0.0, from_left, from_right)

    def _momentum_diagonal_inverse(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # Rhie-Chow 中的 a^{-1} 来自动量方程局部线性系统的对角尺度。
        # 在 PINN 里没有显式 SIMPLE 线性求解器，因此用 FVM 对流/扩散系数
        # 构造一个点态近似；作为插值阻尼系数 detach，避免训练时把该系数也当成
        # 需要反传的额外非线性算子。
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Re = expand_scalar(batch["Re_omega"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])

        ur_e = self._face_plus(ur, dim=1)
        ur_w = self._face_minus(ur, dim=1)
        ut_n = self._face_plus(ut, dim=2, theta_overlap=True)
        ut_s = self._face_minus(ut, dim=2, theta_overlap=True)
        uz_t = self._face_plus(uz, dim=3)
        uz_b = self._face_minus(uz, dim=3)

        conv_diag = (torch.abs(ur_e) + torch.abs(ur_w)) / torch.clamp(dR, min=1e-12)
        conv_diag = conv_diag + (torch.abs(k_theta * ut_n) + torch.abs(k_theta * ut_s)) / torch.clamp(
            dTheta,
            min=1e-12,
        )
        conv_diag = conv_diag + (
            torch.abs(Lambda * Ku * uz_t) + torch.abs(Lambda * Ku * uz_b)
        ) / torch.clamp(dZ, min=1e-12)

        r_e = self._face_plus(r_hat, dim=1)
        r_w = self._face_minus(r_hat, dim=1)
        diff_diag = (r_e + r_w) / (
            torch.clamp(r_hat, min=1e-12) * torch.clamp(dR, min=1e-12) ** 2
        )
        diff_diag = diff_diag + 2.0 * (k_theta ** 2) / torch.clamp(dTheta, min=1e-12) ** 2
        diff_diag = diff_diag + 2.0 * (Lambda ** 2) / torch.clamp(dZ, min=1e-12) ** 2

        diagonal = conv_diag + diff_diag / torch.clamp(Re, min=1e-12) + self.momentum_diagonal_floor
        return torch.reciprocal(torch.clamp(diagonal, min=1e-8)).detach()

    def _rhie_chow_face_velocities(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        p: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # 同位网格上使用 Rhie-Chow 面速度，抑制压力棋盘格。
        # 面压力梯度 = 相邻节点直接差分；节点压力梯度均值 = 两端中心差分均值。
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        k_theta = batch["K_theta"]
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        Eu = expand_scalar(batch["Eu_omega"])

        a_inv = self._momentum_diagonal_inverse(ur, ut, uz, batch)
        coeff = self.rhie_chow_strength * Eu

        p_e = self._neighbor_plus_overlap(p, dim=1, theta_overlap=False)
        p_w = self._neighbor_minus_overlap(p, dim=1, theta_overlap=False)
        p_n = self._neighbor_plus_overlap(p, dim=2, theta_overlap=True)
        p_s = self._neighbor_minus_overlap(p, dim=2, theta_overlap=True)
        p_t = self._neighbor_plus_overlap(p, dim=3, theta_overlap=False)
        p_b = self._neighbor_minus_overlap(p, dim=3, theta_overlap=False)

        grad_r = self.d1(p, dim=1, spacing=dR, periodic=False)
        grad_theta = k_theta * self.d1(p, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        grad_z = (Lambda / torch.clamp(Ku, min=1e-12)) * self.d1(p, dim=3, spacing=dZ, periodic=False)

        a_e = self._face_plus(a_inv, dim=1)
        a_w = self._face_minus(a_inv, dim=1)
        a_n = self._face_plus(a_inv, dim=2, theta_overlap=True)
        a_s = self._face_minus(a_inv, dim=2, theta_overlap=True)
        a_t = self._face_plus(a_inv, dim=3)
        a_b = self._face_minus(a_inv, dim=3)

        direct_e = (p_e - p) / torch.clamp(dR, min=1e-12)
        direct_w = (p - p_w) / torch.clamp(dR, min=1e-12)
        direct_n = k_theta * (p_n - p) / torch.clamp(dTheta, min=1e-12)
        direct_s = k_theta * (p - p_s) / torch.clamp(dTheta, min=1e-12)
        direct_t = (Lambda / torch.clamp(Ku, min=1e-12)) * (p_t - p) / torch.clamp(dZ, min=1e-12)
        direct_b = (Lambda / torch.clamp(Ku, min=1e-12)) * (p - p_b) / torch.clamp(dZ, min=1e-12)

        avg_e = self._face_plus(grad_r, dim=1)
        avg_w = self._face_minus(grad_r, dim=1)
        avg_n = self._face_plus(grad_theta, dim=2, theta_overlap=True)
        avg_s = self._face_minus(grad_theta, dim=2, theta_overlap=True)
        avg_t = self._face_plus(grad_z, dim=3)
        avg_b = self._face_minus(grad_z, dim=3)

        return {
            "ur_e": self._face_plus(ur, dim=1) - coeff * a_e * (direct_e - avg_e),
            "ur_w": self._face_minus(ur, dim=1) - coeff * a_w * (direct_w - avg_w),
            "ut_n": self._face_plus(ut, dim=2, theta_overlap=True) - coeff * a_n * (direct_n - avg_n),
            "ut_s": self._face_minus(ut, dim=2, theta_overlap=True) - coeff * a_s * (direct_s - avg_s),
            "uz_t": self._face_plus(uz, dim=3) - coeff * a_t * (direct_t - avg_t),
            "uz_b": self._face_minus(uz, dim=3) - coeff * a_b * (direct_b - avg_b),
        }

    def conservative_convection(
        self,
        transported: torch.Tensor,
        face_velocity: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # PDF Eq. (2.60)-(2.61): 对流项按控制体面通量离散。
        # 负责输运的速度使用 Rhie-Chow 面速度；被输运量用一阶迎风面值。
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        r_e = self._face_plus(r_hat, dim=1)
        r_w = self._face_minus(r_hat, dim=1)

        if self.convection_interpolation == "central":
            phi_e = self._face_plus(transported, dim=1)
            phi_w = self._face_minus(transported, dim=1)
            phi_n = self._face_plus(transported, dim=2, theta_overlap=True)
            phi_s = self._face_minus(transported, dim=2, theta_overlap=True)
            phi_t = self._face_plus(transported, dim=3)
            phi_b = self._face_minus(transported, dim=3)
        elif self.convection_interpolation == "upwind":
            phi_e = self._upwind_plus(transported, face_velocity["ur_e"], dim=1)
            phi_w = self._upwind_minus(transported, face_velocity["ur_w"], dim=1)
            phi_n = self._upwind_plus(transported, face_velocity["ut_n"], dim=2, theta_overlap=True)
            phi_s = self._upwind_minus(transported, face_velocity["ut_s"], dim=2, theta_overlap=True)
            phi_t = self._upwind_plus(transported, face_velocity["uz_t"], dim=3)
            phi_b = self._upwind_minus(transported, face_velocity["uz_b"], dim=3)
        else:
            phi_e = self._upwind2_plus(transported, face_velocity["ur_e"], dim=1)
            phi_w = self._upwind2_minus(transported, face_velocity["ur_w"], dim=1)
            phi_n = self._upwind2_plus(transported, face_velocity["ut_n"], dim=2, theta_overlap=True)
            phi_s = self._upwind2_minus(transported, face_velocity["ut_s"], dim=2, theta_overlap=True)
            phi_t = self._upwind2_plus(transported, face_velocity["uz_t"], dim=3)
            phi_b = self._upwind2_minus(transported, face_velocity["uz_b"], dim=3)

        radial = (r_e * face_velocity["ur_e"] * phi_e - r_w * face_velocity["ur_w"] * phi_w) / (
            torch.clamp(r_hat, min=1e-12) * torch.clamp(dR, min=1e-12)
        )
        theta = k_theta * (face_velocity["ut_n"] * phi_n - face_velocity["ut_s"] * phi_s) / torch.clamp(
            dTheta,
            min=1e-12,
        )
        axial = Lambda * Ku * (face_velocity["uz_t"] * phi_t - face_velocity["uz_b"] * phi_b) / torch.clamp(
            dZ,
            min=1e-12,
        )
        return radial + theta + axial

    def fvm_laplacian(self, field: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # PDF Eq. (2.66): 扩散项使用有限体积通量形式，而不是先拆成点态二阶中心差分。
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])

        field_e = self._neighbor_plus_overlap(field, dim=1, theta_overlap=False)
        field_w = self._neighbor_minus_overlap(field, dim=1, theta_overlap=False)
        field_n = self._neighbor_plus_overlap(field, dim=2, theta_overlap=True)
        field_s = self._neighbor_minus_overlap(field, dim=2, theta_overlap=True)
        field_t = self._neighbor_plus_overlap(field, dim=3, theta_overlap=False)
        field_b = self._neighbor_minus_overlap(field, dim=3, theta_overlap=False)

        r_e = self._face_plus(r_hat, dim=1)
        r_w = self._face_minus(r_hat, dim=1)
        grad_e = (field_e - field) / torch.clamp(dR, min=1e-12)
        grad_w = (field - field_w) / torch.clamp(dR, min=1e-12)
        grad_n = (field_n - field) / torch.clamp(dTheta, min=1e-12)
        grad_s = (field - field_s) / torch.clamp(dTheta, min=1e-12)
        grad_t = (field_t - field) / torch.clamp(dZ, min=1e-12)
        grad_b = (field - field_b) / torch.clamp(dZ, min=1e-12)

        radial = (r_e * grad_e - r_w * grad_w) / (torch.clamp(r_hat, min=1e-12) * torch.clamp(dR, min=1e-12))
        theta = (k_theta ** 2) * (grad_n - grad_s) / torch.clamp(dTheta, min=1e-12)
        axial = (Lambda ** 2) * (grad_t - grad_b) / torch.clamp(dZ, min=1e-12)
        return radial + theta + axial

    def fvm_rhie_chow_residuals(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # PDF Eq. (2.53)-(2.66): 所有控制方程都在每个格点对应控制体上用净通量描述。
        ur = pred["UR"]
        ut = pred["UT"]
        uz = pred["UZ"]
        p = pred["P"]

        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Eu = expand_scalar(batch["Eu_omega"])
        Re = expand_scalar(batch["Re_omega"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        delta = expand_scalar(batch["delta"])
        sgn_omega = expand_scalar(batch["sgn_omega"])
        g_star = expand_scalar(batch["g_star"])
        absolute_frame = expand_scalar(batch["absolute_frame"])

        face_velocity = self._rhie_chow_face_velocities(ur, ut, uz, p, batch)
        r_e = self._face_plus(r_hat, dim=1)
        r_w = self._face_minus(r_hat, dim=1)

        residual_c = (r_e * face_velocity["ur_e"] - r_w * face_velocity["ur_w"]) / (
            torch.clamp(r_hat, min=1e-12) * torch.clamp(dR, min=1e-12)
        )
        residual_c = residual_c + k_theta * (
            face_velocity["ut_n"] - face_velocity["ut_s"]
        ) / torch.clamp(dTheta, min=1e-12)
        residual_c = residual_c + Lambda * Ku * (
            face_velocity["uz_t"] - face_velocity["uz_b"]
        ) / torch.clamp(dZ, min=1e-12)

        dTheta_ur = self.d1(ur, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dTheta_ut = self.d1(ut, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dR_p = self.d1(p, dim=1, spacing=dR, periodic=False)
        dTheta_p = self.d1(p, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_p = self.d1(p, dim=3, spacing=dZ, periodic=False)

        lap_ur = self.fvm_laplacian(ur, batch)
        lap_ut = self.fvm_laplacian(ut, batch)
        lap_uz = self.fvm_laplacian(uz, batch)

        residual_r = self.conservative_convection(ur, face_velocity, batch)
        residual_r = residual_r - (ut ** 2) / torch.clamp(r_hat, min=1e-12)
        residual_r = residual_r + Eu * dR_p
        residual_r = residual_r - (
            lap_ur
            - ur / torch.clamp(r_hat ** 2, min=1e-12)
            - (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ut
        ) / torch.clamp(Re, min=1e-12)

        residual_theta = self.conservative_convection(ut, face_velocity, batch)
        residual_theta = residual_theta + (ur * ut) / torch.clamp(r_hat, min=1e-12)
        residual_theta = residual_theta + k_theta * Eu * dTheta_p
        residual_theta = residual_theta - (
            lap_ut
            - ut / torch.clamp(r_hat ** 2, min=1e-12)
            + (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ur
        ) / torch.clamp(Re, min=1e-12)

        residual_z = self.conservative_convection(uz, face_velocity, batch)
        residual_z = residual_z + (Lambda / torch.clamp(Ku, min=1e-12)) * Eu * dZ_p
        residual_z = residual_z - lap_uz / torch.clamp(Re, min=1e-12) + g_star

        rotating_weight = 1.0 - absolute_frame
        residual_r = residual_r + rotating_weight * (-delta ** 2 * r_hat + 2.0 * sgn_omega * delta * ut)
        residual_theta = residual_theta + rotating_weight * (-2.0 * sgn_omega * delta * ur)
        return residual_c, residual_r, residual_theta, residual_z

    def dimensionless_laplacian(
        self,
        field: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # 无量纲拉普拉斯算子：
        # dRR + (1/r_hat)dR + K_theta^2 dThetaTheta + Lambda^2 dZZ
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])

        dR_1 = self.d1(field, dim=1, spacing=dR, periodic=False)
        dR_2 = self.d2(field, dim=1, spacing=dR, periodic=False)
        dTheta_2 = self.d2(field, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_2 = self.d2(field, dim=3, spacing=dZ, periodic=False)

        return dR_2 + dR_1 / torch.clamp(r_hat, min=1e-12) + (k_theta ** 2) * dTheta_2 + (Lambda ** 2) * dZ_2

    def outlet_flow_rate_hat(self, uz: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # 这里先计算无量纲出口单流道流量：
        # q_hat = ∫ UZ * r_hat dR dTheta
        # 注意当前 R / Theta 都是含端点网格，因此积分必须用梯形权重。
        phi = batch["phi"]
        r_hat = batch["r_hat"]
        # 这里是在出口截面上做二维积分，权重张量应为 [B, R, Theta]。
        # 如果沿用四维广播形状，会多挤出一维，最终把单个流量错误地扩成一整排数。
        dR = batch["dR"].view(-1, 1, 1)
        dTheta = batch["dTheta"].view(-1, 1, 1)

        r_weight = line_quadrature_weight(uz.shape[1], uz.device, uz.dtype)
        theta_weight = line_quadrature_weight(uz.shape[2], uz.device, uz.dtype)
        quad_weight = r_weight.view(1, -1, 1) * theta_weight.view(1, 1, -1)

        outlet_weight = phi[:, :, :, -1] * r_hat[:, :, :, -1] * quad_weight * dR * dTheta
        return torch.sum(uz[:, :, :, -1] * outlet_weight, dim=(1, 2))

    def outlet_flow_rate(self, uz: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # 把无量纲流量再映回物理空间，便于后处理打印。
        delta_r = batch["delta_r"].view(-1)
        theta0 = batch["theta0"].view(-1)
        u_zo = batch["u_zo"].view(-1)
        return self.outlet_flow_rate_hat(uz, batch) * u_zo * (delta_r ** 2) * theta0

    def theta_periodic_error(self, field: torch.Tensor) -> torch.Tensor:
        seam = field[:, :, 0, :] - field[:, :, -1, :]
        return torch.mean(seam ** 2)

    def blade_noslip_error(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        # 叶片区硬约束的目标是：
        # UR=0, UZ=0, UT=solid_ut
        solid = 1.0 - batch["phi"]
        loss_ur = self.weighted_mse(pred["UR"], solid)
        loss_ut = self.weighted_mse(pred["UT"] - batch["solid_ut"], solid)
        loss_uz = self.weighted_mse(pred["UZ"], solid)
        return loss_ur + loss_ut + loss_uz

    def residual_fields(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ur = pred["UR"]
        ut = pred["UT"]
        uz = pred["UZ"]
        p = pred["P"]

        if self.physics_discretization == "fvm_rhie_chow":
            return self.fvm_rhie_chow_residuals(pred, batch)

        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Eu = expand_scalar(batch["Eu_omega"])
        Re = torch.clamp(expand_scalar(batch["Re_omega"]), min=1e-12)
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        delta = expand_scalar(batch["delta"])
        sgn_omega = expand_scalar(batch["sgn_omega"])
        g_star = expand_scalar(batch["g_star"])
        absolute_frame = expand_scalar(batch["absolute_frame"])

        dR_ur = self.d1(ur, dim=1, spacing=dR, periodic=False)
        dTheta_ur = self.d1(ur, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_ur = self.d1(ur, dim=3, spacing=dZ, periodic=False)

        dR_ut = self.d1(ut, dim=1, spacing=dR, periodic=False)
        dTheta_ut = self.d1(ut, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_ut = self.d1(ut, dim=3, spacing=dZ, periodic=False)

        dR_uz = self.d1(uz, dim=1, spacing=dR, periodic=False)
        dTheta_uz = self.d1(uz, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_uz = self.d1(uz, dim=3, spacing=dZ, periodic=False)

        dR_p = self.d1(p, dim=1, spacing=dR, periodic=False)
        dTheta_p = self.d1(p, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        dZ_p = self.d1(p, dim=3, spacing=dZ, periodic=False)

        lap_ur = self.dimensionless_laplacian(ur, batch)
        lap_ut = self.dimensionless_laplacian(ut, batch)
        lap_uz = self.dimensionless_laplacian(uz, batch)

        residual_c = self.d1(r_hat * ur, dim=1, spacing=dR, periodic=False) / torch.clamp(r_hat, min=1e-12)
        residual_c = residual_c + k_theta * dTheta_ut + Lambda * Ku * dZ_uz

        residual_r = ur * dR_ur + k_theta * ut * dTheta_ur + Lambda * Ku * uz * dZ_ur
        residual_r = residual_r - (ut ** 2) / torch.clamp(r_hat, min=1e-12)
        residual_r = residual_r + Eu * dR_p
        residual_r = residual_r - (
            lap_ur
            - ur / torch.clamp(r_hat ** 2, min=1e-12)
            - (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ut
        ) / Re

        residual_theta = ur * dR_ut + k_theta * ut * dTheta_ut + Lambda * Ku * uz * dZ_ut
        residual_theta = residual_theta + (ur * ut) / torch.clamp(r_hat, min=1e-12)
        residual_theta = residual_theta + k_theta * Eu * dTheta_p
        residual_theta = residual_theta - (
            lap_ut
            - ut / torch.clamp(r_hat ** 2, min=1e-12)
            + (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ur
        ) / Re

        residual_z = ur * dR_uz + k_theta * ut * dTheta_uz + Lambda * Ku * uz * dZ_uz
        residual_z = residual_z + (Lambda / torch.clamp(Ku, min=1e-12)) * Eu * dZ_p
        residual_z = residual_z - lap_uz / Re + g_star

        rotating_weight = 1.0 - absolute_frame
        residual_r = residual_r + rotating_weight * (-delta ** 2 * r_hat + 2.0 * sgn_omega * delta * ut)
        residual_theta = residual_theta + rotating_weight * (-2.0 * sgn_omega * delta * ur)
        return residual_c, residual_r, residual_theta, residual_z

    def forward(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ur = pred["UR"]
        ut = pred["UT"]
        uz = pred["UZ"]
        p = pred["P"]

        phi = batch["phi"]
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        pde_mask = self.build_pde_mask(phi)

        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Eu = expand_scalar(batch["Eu_omega"])
        Re = expand_scalar(batch["Re_omega"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])
        delta = expand_scalar(batch["delta"])
        sgn_omega = expand_scalar(batch["sgn_omega"])
        g_star = expand_scalar(batch["g_star"])
        absolute_frame = expand_scalar(batch["absolute_frame"])

        if self.physics_discretization == "fvm_rhie_chow":
            residual_c, residual_r, residual_theta, residual_z = self.fvm_rhie_chow_residuals(pred, batch)
        else:
            # 所有 Theta 导数都必须使用“首尾重合网格”的专用差分。
            dR_ur = self.d1(ur, dim=1, spacing=dR, periodic=False)
            dTheta_ur = self.d1(ur, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
            dZ_ur = self.d1(ur, dim=3, spacing=dZ, periodic=False)

            dR_ut = self.d1(ut, dim=1, spacing=dR, periodic=False)
            dTheta_ut = self.d1(ut, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
            dZ_ut = self.d1(ut, dim=3, spacing=dZ, periodic=False)

            dR_uz = self.d1(uz, dim=1, spacing=dR, periodic=False)
            dTheta_uz = self.d1(uz, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
            dZ_uz = self.d1(uz, dim=3, spacing=dZ, periodic=False)

            dR_p = self.d1(p, dim=1, spacing=dR, periodic=False)
            dTheta_p = self.d1(p, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
            dZ_p = self.d1(p, dim=3, spacing=dZ, periodic=False)

            lap_ur = self.dimensionless_laplacian(ur, batch)
            lap_ut = self.dimensionless_laplacian(ut, batch)
            lap_uz = self.dimensionless_laplacian(uz, batch)

            # 连续方程残差：
            # Rc = 1/r_hat dR(r_hat UR) + K_theta dTheta UT + Lambda Ku dZ UZ
            residual_c = self.d1(r_hat * ur, dim=1, spacing=dR, periodic=False) / torch.clamp(r_hat, min=1e-12)
            residual_c = residual_c + k_theta * dTheta_ut + Lambda * Ku * dZ_uz

            # 径向动量残差。
            residual_r = ur * dR_ur + k_theta * ut * dTheta_ur + Lambda * Ku * uz * dZ_ur
            residual_r = residual_r - (ut ** 2) / torch.clamp(r_hat, min=1e-12)
            residual_r = residual_r + Eu * dR_p
            residual_r = residual_r - (
                lap_ur
                - ur / torch.clamp(r_hat ** 2, min=1e-12)
                - (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ut
            ) / Re

            # 周向动量残差。
            residual_theta = ur * dR_ut + k_theta * ut * dTheta_ut + Lambda * Ku * uz * dZ_ut
            residual_theta = residual_theta + (ur * ut) / torch.clamp(r_hat, min=1e-12)
            residual_theta = residual_theta + k_theta * Eu * dTheta_p
            residual_theta = residual_theta - (
                lap_ut
                - ut / torch.clamp(r_hat ** 2, min=1e-12)
                + (2.0 * k_theta / torch.clamp(r_hat, min=1e-12)) * dTheta_ur
            ) / Re

            # 轴向动量残差。
            residual_z = ur * dR_uz + k_theta * ut * dTheta_uz + Lambda * Ku * uz * dZ_uz
            residual_z = residual_z + (Lambda / torch.clamp(Ku, min=1e-12)) * Eu * dZ_p
            residual_z = residual_z - lap_uz / Re + g_star

            # 绝对参考系默认不需要旋转附加项。
            # 如果切换到旋转参考系，就把对应项加回来。
            rotating_weight = 1.0 - absolute_frame
            residual_r = residual_r + rotating_weight * (-delta ** 2 * r_hat + 2.0 * sgn_omega * delta * ut)
            residual_theta = residual_theta + rotating_weight * (-2.0 * sgn_omega * delta * ur)

        loss_c = self.weighted_mse(residual_c, pde_mask)
        loss_r = self.weighted_mse(residual_r, pde_mask)
        loss_theta = self.weighted_mse(residual_theta, pde_mask)
        loss_z = self.weighted_mse(residual_z, pde_mask)
        with torch.no_grad():
            scaled_residuals = self.scaled_residual_metrics(
                pred,
                batch,
                residual_c,
                residual_r,
                residual_theta,
                residual_z,
                pde_mask,
            )

        # 出口流量损失使用无量纲形式，这样和其余 PDE 残差的量纲更一致。
        q_hat_pred = self.outlet_flow_rate_hat(uz, batch)
        q_hat_target = batch["qv_hat"]
        loss_qv = torch.mean((q_hat_pred - q_hat_target) ** 2)
        pressure_highpass = self.pressure_highpass(p)
        loss_p_highpass = self.weighted_mse(pressure_highpass, pde_mask)
        p_mean = torch.sum(p * pde_mask) / torch.clamp(torch.sum(pde_mask), min=1e-12)
        loss_p_energy = self.weighted_mse(p - p_mean, pde_mask)
        loss_p_highpass_ratio = loss_p_highpass / torch.clamp(loss_p_energy, min=1e-12)
        loss_p_highpass_penalty = loss_p_highpass_ratio if self.pressure_highpass_normalized else loss_p_highpass

        # 下面两个量只是用于检查边界条件是否真的被硬约束住。
        loss_bc_periodic = (
            self.theta_periodic_error(ur)
            + self.theta_periodic_error(ut)
            + self.theta_periodic_error(uz)
            + self.theta_periodic_error(p)
        )
        loss_bc_blade = self.blade_noslip_error(pred, batch)

        total = loss_c + loss_r + loss_theta + loss_z + loss_qv
        total = total + self.pressure_highpass_weight * loss_p_highpass_penalty
        return total, {
            "loss_c": loss_c,
            "loss_r": loss_r,
            "loss_theta": loss_theta,
            "loss_z": loss_z,
            "loss_qv": loss_qv,
            "loss_p_highpass": loss_p_highpass,
            "loss_p_highpass_ratio": loss_p_highpass_ratio,
            "loss_p_highpass_penalty": loss_p_highpass_penalty,
            "loss_bc_periodic": loss_bc_periodic,
            "loss_bc_blade": loss_bc_blade,
            "loss_phys": total,
            **scaled_residuals,
            "q_hat_pred": torch.mean(q_hat_pred),
            "q_hat_target": torch.mean(q_hat_target),
            "rhie_chow_strength": torch.tensor(
                self.rhie_chow_strength,
                device=total.device,
                dtype=total.dtype,
            ),
        }




class SurrogateModeling:
    # 训练器同时支持两种模式：
    # 1. 有监督样本：数据损失 + 物理损失
    # 2. 纯物理调试：仅依靠物理损失检查流程是否能跑通
    def __init__(
        self,
        train_cases: Sequence[Mapping[str, Any]],
        val_cases: Sequence[Mapping[str, Any]] | None = None,
        *,
        input_mode: str = "both",
        batch_size: int = 2,
        lr: float = 1e-3,
        modes: int = 12,
        width: int = 32,
        depth: int = 4,
        z_padding: int = 8,
        operator_variant: str = "hf_cfno",
        high_modes: int | None = None,
        fourier_feature_bands: Sequence[int] = (1, 2, 4, 8),
        hf_high_gate_init: float = -1.0,
        hf_use_local_highpass: bool = True,
        pressure_smoothing: float = 0.0,
        pressure_reference_mode: str = "origin",
        pressure_supervision_mode: str = "gradient",
        pressure_data_reference: str = "training_origin",
        pressure_highpass_weight: float = 0.0,
        pressure_highpass_normalized: bool = True,
        physics_discretization: str = "fvm_rhie_chow",
        rhie_chow_strength: float = 0.35,
        momentum_diagonal_floor: float = 1.0,
        convection_interpolation: str = "central",
        fluent_continuity_first5: Sequence[float] | str | None = None,
        fluent_continuity_scale: float | None = None,
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        warmup_epochs: int = 20,
        ramp_epochs: int = 30,
        use_kkt_projection: bool = False,    # KKT投影疑似没写对，就不放出来了
        kkt_projection_iters: int = 24,
        kkt_projection_strength: float = 0.35,
        learn_ibm_params: bool = True,
        ibm_c_range: tuple[float, float] = (0.25, 4.0),
        ibm_epsilon_range: tuple[float, float] = (0.01, 0.08),
        ibm_hidden: int = 16,
        micro_batch_size: int | None = None,
        slice_batch_size: int | None = None,
        auto_cuda_batching: bool = True,
        cuda_memory_fraction: float = 0.65,
        use_activation_checkpointing: bool = True,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.micro_batch_size = None if micro_batch_size is None else max(1, int(micro_batch_size))
        self.auto_cuda_batching = bool(auto_cuda_batching)
        self.cuda_memory_fraction = float(np.clip(cuda_memory_fraction, 0.05, 0.95))
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self._runtime_micro_batch_size: int | None = self.micro_batch_size
        self.pressure_supervision_mode = str(pressure_supervision_mode).lower()
        if self.pressure_supervision_mode not in {"gradient", "value", "none"}:
            raise ValueError("pressure_supervision_mode must be 'gradient', 'value', or 'none'.")
        self.pressure_reference_mode = str(pressure_reference_mode).lower()
        if self.pressure_reference_mode not in {"origin", "none"}:
            raise ValueError("pressure_reference_mode must be 'origin' or 'none'.")
        self.pressure_data_reference = str(pressure_data_reference).lower()
        if self.pressure_data_reference not in {"training_origin", "absolute"}:
            raise ValueError("pressure_data_reference must be 'training_origin' or 'absolute'.")
        convection_interpolation = _normalize_convection_interpolation(convection_interpolation)
        self.train_dataset = BladeFlowDataset(
            train_cases,
            input_mode=input_mode,
            pressure_reference=self.pressure_data_reference,
        )
        self.val_dataset = BladeFlowDataset(
            val_cases if val_cases is not None else train_cases,
            input_mode=input_mode,
            pressure_reference=self.pressure_data_reference,
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        input_channels = self.train_dataset[0]["x"].shape[0]
        self.model = SliceWiseFNOFlowModel(
            input_channels=input_channels,
            modes=modes,
            width=width,
            depth=depth,
            z_padding=z_padding,
            operator_variant=operator_variant,
            high_modes=high_modes,
            fourier_feature_bands=fourier_feature_bands,
            hf_high_gate_init=hf_high_gate_init,
            hf_use_local_highpass=hf_use_local_highpass,
            pressure_smoothing=pressure_smoothing,
            pressure_reference_mode=self.pressure_reference_mode,
            slice_batch_size=slice_batch_size,
            auto_cuda_batching=auto_cuda_batching,
            cuda_memory_fraction=cuda_memory_fraction,
            use_activation_checkpointing=use_activation_checkpointing,
        ).to(self.device)
        self.model_config = {
            "input_channels": input_channels,
            "modes": modes,
            "width": width,
            "depth": depth,
            "z_padding": z_padding,
            "operator_variant": str(operator_variant),
            "high_modes": high_modes,
            "fourier_feature_bands": tuple(int(v) for v in fourier_feature_bands),
            "hf_high_gate_init": float(hf_high_gate_init),
            "hf_use_local_highpass": bool(hf_use_local_highpass),
            "pressure_smoothing": float(np.clip(pressure_smoothing, 0.0, 1.0)),
            "pressure_reference_mode": self.pressure_reference_mode,
            "pressure_supervision_mode": self.pressure_supervision_mode,
            "pressure_data_reference": self.pressure_data_reference,
            "slice_batch_size": slice_batch_size,
            "auto_cuda_batching": bool(auto_cuda_batching),
            "cuda_memory_fraction": self.cuda_memory_fraction,
            "use_activation_checkpointing": self.use_activation_checkpointing,
            "output_channels": 4,
        }
        default_ibm_c = float(
            torch.stack([sample["ibm_C"].reshape(-1).mean() for sample in self.train_dataset.samples]).mean().item()
        )
        default_ibm_epsilon = float(
            torch.stack([sample["ibm_epsilon"].reshape(-1).mean() for sample in self.train_dataset.samples]).mean().item()
        )
        self.learn_ibm_params = bool(learn_ibm_params)
        self.ibm_mask_controller = (
            AdaptiveIBMMaskController(
                c_range=ibm_c_range,
                epsilon_range=ibm_epsilon_range,
                hidden=ibm_hidden,
                default_c=default_ibm_c,
                default_epsilon=default_ibm_epsilon,
            ).to(self.device)
            if self.learn_ibm_params
            else None
        )
        self.ibm_config = {
            "learn_ibm_params": self.learn_ibm_params,
            "ibm_profile_mode": "span",
            "ibm_c_range": tuple(float(v) for v in ibm_c_range),
            "ibm_epsilon_range": tuple(float(v) for v in ibm_epsilon_range),
            "ibm_hidden": int(ibm_hidden),
            "default_ibm_c": default_ibm_c,
            "default_ibm_epsilon": default_ibm_epsilon,
        }
        self.trainer_config = {
            "input_mode": input_mode,
            "batch_size": batch_size,
            "micro_batch_size": micro_batch_size,
            "slice_batch_size": slice_batch_size,
            "auto_cuda_batching": bool(auto_cuda_batching),
            "cuda_memory_fraction": self.cuda_memory_fraction,
            "use_activation_checkpointing": self.use_activation_checkpointing,
            "lr": lr,
            "operator_variant": str(operator_variant),
            "high_modes": high_modes,
            "fourier_feature_bands": tuple(int(v) for v in fourier_feature_bands),
            "hf_high_gate_init": float(hf_high_gate_init),
            "hf_use_local_highpass": bool(hf_use_local_highpass),
            "pressure_smoothing": float(np.clip(pressure_smoothing, 0.0, 1.0)),
            "pressure_reference_mode": self.pressure_reference_mode,
            "pressure_supervision_mode": self.pressure_supervision_mode,
            "pressure_data_reference": self.pressure_data_reference,
            "pressure_highpass_weight": float(max(pressure_highpass_weight, 0.0)),
            "pressure_highpass_normalized": bool(pressure_highpass_normalized),
            "physics_discretization": str(physics_discretization).lower(),
            "rhie_chow_strength": float(max(rhie_chow_strength, 0.0)),
            "momentum_diagonal_floor": float(max(momentum_diagonal_floor, 1e-8)),
            "convection_interpolation": convection_interpolation,
            "fluent_continuity_first5": list(_float_sequence_from_any(fluent_continuity_first5)),
            "fluent_continuity_scale": _resolve_fluent_continuity_scale(
                fluent_continuity_scale,
                fluent_continuity_first5,
            ),
            "data_weight": data_weight,
            "physics_weight": physics_weight,
            "warmup_epochs": warmup_epochs,
            "ramp_epochs": ramp_epochs,
            "use_kkt_projection": bool(use_kkt_projection),
            "kkt_projection_iters": int(kkt_projection_iters),
            "kkt_projection_strength": float(kkt_projection_strength),
            "learn_ibm_params": self.learn_ibm_params,
            "ibm_profile_mode": "span",
            "ibm_c_range": tuple(float(v) for v in ibm_c_range),
            "ibm_epsilon_range": tuple(float(v) for v in ibm_epsilon_range),
            "ibm_hidden": int(ibm_hidden),
        }
        self.checkpoint_metadata: dict[str, Any] | None = None

        self.physics_loss = BladeFlowPhysicsLoss(
            pressure_highpass_weight=pressure_highpass_weight,
            pressure_highpass_normalized=pressure_highpass_normalized,
            physics_discretization=physics_discretization,
            rhie_chow_strength=rhie_chow_strength,
            momentum_diagonal_floor=momentum_diagonal_floor,
            convection_interpolation=convection_interpolation,
            fluent_continuity_first5=fluent_continuity_first5,
            fluent_continuity_scale=fluent_continuity_scale,
        ).to(self.device)
        self.kkt_projection = build_kkt_projection(
            enabled=bool(use_kkt_projection),
            iterations=int(kkt_projection_iters),
            strength=float(kkt_projection_strength),
            device=self.device,
        )
        optimizer_params: list[nn.Parameter] = list(self.model.parameters())
        if self.ibm_mask_controller is not None:
            optimizer_params.extend(list(self.ibm_mask_controller.parameters()))
        self.optimizer = torch.optim.Adam(optimizer_params, lr=lr)

        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

        self.print_preparation_summary(case_index=0)

    def print_preparation_summary(self, case_index: int = 0) -> None:
        # 打印一次样本准备摘要，方便确认几何、尺度和边界设置是否符合预期。
        sample = self.train_dataset[case_index]
        case = self.train_dataset.cases[case_index]
        blade_params_all = [str(item["blade_params"]) for item in self.train_dataset.cases if "blade_params" in item]
        blade_geometry_count = len(set(blade_params_all))

        blade_cells = int(sample["blade_mask"].sum().item())
        total_cells = int(sample["blade_mask"].numel())
        fluid_ratio = float(sample["phi"].mean().item())
        has_target = bool(sample["has_target"].item() > 0.5)
        ibm_c_mean, ibm_c_min, ibm_c_max = _profile_mean_min_max(sample["ibm_C"])
        ibm_epsilon_mean, ibm_epsilon_min, ibm_epsilon_max = _profile_mean_min_max(sample["ibm_epsilon"])

        print("\n========== SurrogateModeling: 数据准备摘要 ==========")
        print(f"device                : {self.device}")
        print(f"train_cases / val_cases: {len(self.train_dataset)} / {len(self.val_dataset)}")
        print(f"train_blade_geometries: {blade_geometry_count}")
        print(f"input_mode            : {self.train_dataset.input_mode}")
        print(f"operator_variant      : {self.model_config.get('operator_variant', 'cfno')}")
        print(
            "cuda_batching         : "
            f"auto={self.auto_cuda_batching}, "
            f"micro={self.micro_batch_size or 'auto'}, "
            f"slice={self.model.slice_batch_size or 'auto'}, "
            f"checkpoint={self.use_activation_checkpointing}"
        )
        print(f"kkt_projection        : {self.kkt_projection is not None}")
        print(f"learn_ibm_params      : {self.learn_ibm_params}")
        print(f"pressure_supervision  : {self.pressure_supervision_mode}")
        print(f"pressure_reference    : {self.pressure_reference_mode}")
        print(f"pressure_data_ref     : {self.pressure_data_reference}")
        print(f"physics_discretization: {self.physics_loss.physics_discretization}")
        print(
            "rhie_chow            : "
            f"strength={self.physics_loss.rhie_chow_strength:.6g}, "
            f"diag_floor={self.physics_loss.momentum_diagonal_floor:.6g}"
        )
        print(f"convection_interp    : {self.physics_loss.convection_interpolation}")
        fluent_c_scale = self.physics_loss.fluent_continuity_scale
        if fluent_c_scale is not None:
            print(
                "fluent_continuity    : "
                f"first5={list(self.physics_loss.fluent_continuity_first5)}, "
                f"scale={fluent_c_scale:.6g}"
            )
        else:
            print("fluent_continuity    : current mass-flux scale")
        print(
            f"ibm_range             : C in {self.ibm_config['ibm_c_range']}, "
            f"epsilon in {self.ibm_config['ibm_epsilon_range']}"
        )
        print(f"input_channels        : {sample['x'].shape[0]}")
        print("theta_periodic        : 输入拼缝投影 + 输出拼缝投影 + 周期专用差分")
        print("z_boundary            : FNO 前向使用复制填充，弱化 Z 向伪周期影响")
        print(f"grid                  : n = {sample['x'].shape[-1]}")
        print(
            f"geometry              : rh={float(case['rh']):.6g}, "
            f"rs={float(case['rs']):.6g}, h={float(case['h']):.6g}, "
            f"n_blade={int(case['n_blade'])}"
        )
        print(
            f"work_condition        : omega={float(case['omega']):.6g}, "
            f"qv_total={float(case['qv']):.6g}, "
            f"qv_passage={float(sample['qv_passage'].item()):.6g}, "
            f"qv_hat={float(sample['qv_hat'].item()):.6g}, "
            f"g={float(case.get('g', 9.8)):.6g}, "
            f"g_star={float(sample['g_star'].item()):.6g}"
        )
        print(
            f"scales                : u_omega={float(sample['u_omega'].item()):.6g}, "
            f"u_zo={float(sample['u_zo'].item()):.6g}, "
            f"P0={float(sample['P0'].item()):.6g}"
        )
        print(
            f"blade / fluid         : blade_cells={blade_cells}, "
            f"total_cells={total_cells}, fluid_phi_mean={fluid_ratio:.6f}"
        )
        print(f"supervised_target     : {'yes' if has_target else 'no, pure physics debug'}")
        print(
            f"default_ibm_profile   : C mean/min/max={ibm_c_mean:.6g}/{ibm_c_min:.6g}/{ibm_c_max:.6g}, "
            f"epsilon mean/min/max={ibm_epsilon_mean:.6g}/{ibm_epsilon_min:.6g}/{ibm_epsilon_max:.6g}"
        )
        if "blade_params" in case:
            print(f"sample_blade_params   : {case['blade_params']}")
        print("===============================================\n")

    def _current_ibm_params(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # 如果开启了自适应 IBM 参数，就按当前样本动态求一对 C / epsilon。
        # 否则退回到样本里保存的默认值。
        if self.ibm_mask_controller is None:
            ibm_c = _expand_ibm_parameter(batch["ibm_C"], batch["signed_distance"])
            ibm_epsilon = _expand_ibm_parameter(batch["ibm_epsilon"], batch["signed_distance"])
            return ibm_c, ibm_epsilon
        return self.ibm_mask_controller(batch)

    def _compose_model_input(self, batch: Mapping[str, torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
        # x 中只有前 1-2 个通道依赖叶片场，其余几何/工况通道都是固定的。
        # 因此这里按当前的 phi 重新拼一次输入张量。
        channels: list[torch.Tensor] = []
        if self.train_dataset.input_mode == "mask":
            channels.append(batch["blade_mask"])
        elif self.train_dataset.input_mode == "phi":
            channels.append(phi)
        elif self.train_dataset.input_mode == "both":
            channels.append(batch["blade_mask"])
            channels.append(phi)
        else:
            raise ValueError("input_mode must be 'mask', 'phi', or 'both'")

        theta_phase = 2.0 * np.pi * batch["Theta"]
        theta_sin = torch.sin(theta_phase)
        theta_cos = torch.cos(theta_phase)
        ones = torch.ones_like(batch["r_hat"])

        channels.extend(
            [
                batch["r_hat"],
                batch["K_theta"],
                theta_sin,
                theta_cos,
                batch["Z"],
                batch["solid_ut"],
                ones * expand_scalar(batch["Eu_omega"]),
                ones * expand_scalar(batch["Re_omega"]),
                ones * expand_scalar(batch["Lambda"]),
                ones * expand_scalar(batch["Ku"]),
                ones * expand_scalar(batch["delta"]),
                ones * expand_scalar(batch["sgn_omega"]),
                ones * expand_scalar(batch["g_star"]),
            ]
        )

        x = torch.stack(channels, dim=1)
        return hard_project_theta_periodic(x, theta_dim=3)

    def _apply_kkt_projection(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return apply_kkt_projection(self.kkt_projection, pred, batch)

    def _prepare_runtime_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 训练和部署阶段都使用“当前学习到的 IBM 参数”实时重建 phi 和 x。
        ibm_c, ibm_epsilon = self._current_ibm_params(batch)
        learned_phi = build_phi_from_signed_distance(batch["signed_distance"], ibm_c, ibm_epsilon)
        learned_phi = torch.where(batch["blade_mask"] > 0.5, torch.zeros_like(learned_phi), learned_phi)
        if "has_true_signed_distance" in batch:
            true_distance_mask = batch["has_true_signed_distance"].view(-1, 1, 1, 1) > 0.5
            case_ibm_c = _expand_ibm_parameter(batch["ibm_C"], batch["signed_distance"])
            case_ibm_epsilon = _expand_ibm_parameter(batch["ibm_epsilon"], batch["signed_distance"])
            phi = torch.where(true_distance_mask, learned_phi, batch["phi"])
            ibm_c = torch.where(true_distance_mask, ibm_c, case_ibm_c)
            ibm_epsilon = torch.where(
                true_distance_mask,
                ibm_epsilon,
                case_ibm_epsilon,
            )
        else:
            phi = learned_phi
        phi = torch.clamp(phi, min=0.0, max=1.0)
        phi = hard_project_theta_periodic(phi, theta_dim=2)

        runtime_batch = dict(batch)
        runtime_batch["phi"] = phi
        runtime_batch["x"] = self._compose_model_input(runtime_batch, phi)
        runtime_batch["ibm_C"] = ibm_c.squeeze(-1).squeeze(-1)
        runtime_batch["ibm_epsilon"] = ibm_epsilon.squeeze(-1).squeeze(-1)
        return runtime_batch

    def _resolve_case_sample(
        self,
        *,
        case_index: int = 0,
        case: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, torch.Tensor]]:
        # 训练集中的样本和部署阶段的外部样本都统一走这个入口。
        if case is None:
            return self.train_dataset.cases[case_index], self.train_dataset[case_index]

        dataset = BladeFlowDataset(
            [case],
            input_mode=self.train_dataset.input_mode,
            pressure_reference=self.pressure_data_reference,
        )
        return case, dataset[0]

    def _build_boundary_for_case(
        self,
        case: Mapping[str, Any],
    ):
        # 三维转子图优先直接使用 blade_params 重建几何。
        blade_params = _pick(case, "blade_params")
        if blade_params is None:
            return None

        config = FlowCaseConfig.from_mapping(case)
        return build_blade_boundary(blade_params, config.make_passage_geometry())

    @torch.no_grad()
    def _predict_case_bundle(
        self,
        *,
        case_index: int = 0,
        case: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 统一构造部署所需的整套对象，避免 post 阶段重复 build dataset / 重复 forward。
        case_data, sample = self._resolve_case_sample(case_index=case_index, case=case)
        self.model.eval()
        if self.ibm_mask_controller is not None:
            self.ibm_mask_controller.eval()
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
        batch = self._prepare_runtime_batch(batch)
        pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])
        pred = self._apply_kkt_projection(pred, batch)
        pred_dim = {key: value[0].detach().cpu() for key, value in pred.items()}
        sample_runtime = dict(sample)
        sample_runtime["x"] = batch["x"][0].detach().cpu()
        sample_runtime["phi"] = batch["phi"][0].detach().cpu()
        sample_runtime["ibm_C"] = batch["ibm_C"][0].detach().cpu()
        sample_runtime["ibm_epsilon"] = batch["ibm_epsilon"][0].detach().cpu()

        u_omega = float(batch["u_omega"][0].cpu().item())
        u_zo = float(batch["u_zo"][0].cpu().item())
        p0 = float(batch["P0"][0].cpu().item())
        pred_phy = {
            "UR": pred_dim["UR"] * u_omega,
            "UT": pred_dim["UT"] * u_omega,
            "UZ": pred_dim["UZ"] * u_zo,
            "P": pred_dim["P"] * p0,
        }

        boundary = self._build_boundary_for_case(case_data)
        return {
            "case": case_data,
            "sample": sample_runtime,
            "batch": batch,
            "pred_dim": pred_dim,
            "pred_phy": pred_phy,
            "boundary": boundary,
        }

    _trace_streamline_cylindrical = _trace_streamline_cylindrical_impl
    plot_3d_streamlines = _plot_3d_streamlines_impl
    plot_blade_spans = _plot_blade_spans_impl
    plot_training_history = _plot_training_history_impl
    _dct_along = staticmethod(_dct_along_impl)
    _axis_spectrum_energy = staticmethod(_axis_spectrum_energy_impl)
    frequency_energy_diagnostics = _frequency_energy_diagnostics_impl
    plot_frequency_energy_trends = _plot_frequency_energy_trends_impl
    plot_cfd_spans = _plot_cfd_spans_impl
    compare_cfd_prediction_spans = _compare_cfd_prediction_spans_impl
    plot_local_physics_residual_spans = _plot_local_physics_residual_spans_impl
    post_process_case = _post_process_case_impl





    @classmethod
    def build_pure_physics_debug_trainer(
        cls,
        *,
        blade_params: str | Path | Sequence[str | Path],
        n: int = 64,
        rh: float | None = None,
        rs: float | None = None,
        h: float | None = None,
        mu: float = 0.006,
        rho: float = 10650.0,
        omega: float = -210.0 * 2 * np.pi / 60.0,
        qv: float = 0.16,
        n_blade: int | None = None,
        z0: float | None = None,
        g: float = 9.8,
        input_mode: str = "both",
        batch_size: int = 1,
        lr: float = 1e-3,
        modes: int = 12,
        width: int = 32,
        depth: int = 4,
        z_padding: int = 8,
        operator_variant: str = "hf_cfno",
        high_modes: int | None = None,
        fourier_feature_bands: Sequence[int] = (1, 2, 4, 8),
        hf_high_gate_init: float = -1.0,
        hf_use_local_highpass: bool = True,
        pressure_smoothing: float = 0.0,
        pressure_highpass_weight: float = 0.0,
        pressure_highpass_normalized: bool = True,
        physics_discretization: str = "fvm_rhie_chow",
        rhie_chow_strength: float = 0.35,
        momentum_diagonal_floor: float = 1.0,
        convection_interpolation: str = "central",
        use_kkt_projection: bool = False,
        kkt_projection_iters: int = 24,
        kkt_projection_strength: float = 0.35,
        learn_ibm_params: bool = True,
        ibm_c_range: tuple[float, float] = (0.25, 4.0),
        ibm_epsilon_range: tuple[float, float] = (0.01, 0.08),
        ibm_hidden: int = 16,
        micro_batch_size: int | None = None,
        slice_batch_size: int | None = None,
        auto_cuda_batching: bool = True,
        cuda_memory_fraction: float = 0.65,
        use_activation_checkpointing: bool = True,
        device: str = "cuda",
        val_blade_params: str | Path | Sequence[str | Path] | None = None,
    ) -> "SurrogateModeling":
        train_cases = make_pure_physics_debug_cases(
            blade_params=blade_params,
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
        val_cases = None
        if val_blade_params is not None:
            val_cases = make_pure_physics_debug_cases(
                blade_params=val_blade_params,
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
        return cls(
            train_cases=train_cases,
            val_cases=val_cases if val_cases is not None else train_cases,
            input_mode=input_mode,
            batch_size=batch_size,
            lr=lr,
            modes=modes,
            width=width,
            depth=depth,
            z_padding=z_padding,
            operator_variant=operator_variant,
            high_modes=high_modes,
            fourier_feature_bands=fourier_feature_bands,
            hf_high_gate_init=hf_high_gate_init,
            hf_use_local_highpass=hf_use_local_highpass,
            pressure_smoothing=pressure_smoothing,
            pressure_highpass_weight=pressure_highpass_weight,
            pressure_highpass_normalized=pressure_highpass_normalized,
            physics_discretization=physics_discretization,
            rhie_chow_strength=rhie_chow_strength,
            momentum_diagonal_floor=momentum_diagonal_floor,
            convection_interpolation=convection_interpolation,
            use_kkt_projection=use_kkt_projection,
            kkt_projection_iters=kkt_projection_iters,
            kkt_projection_strength=kkt_projection_strength,
            data_weight=0.0,
            physics_weight=1.0,
            warmup_epochs=0,
            ramp_epochs=0,
            learn_ibm_params=learn_ibm_params,
            ibm_c_range=ibm_c_range,
            ibm_epsilon_range=ibm_epsilon_range,
            ibm_hidden=ibm_hidden,
            micro_batch_size=micro_batch_size,
            slice_batch_size=slice_batch_size,
            auto_cuda_batching=auto_cuda_batching,
            cuda_memory_fraction=cuda_memory_fraction,
            use_activation_checkpointing=use_activation_checkpointing,
            device=device,
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    def supervised_pressure_gradient_loss(
        self,
        pred_p: torch.Tensor,
        target_p: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # CFD 压力的绝对零点没有监督意义；这里只比较压力梯度。
        # 梯度采用和物理残差一致的无量纲柱坐标缩放：
        # dP/dR, K_theta*dP/dTheta, Lambda*dP/dZ。
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        k_theta = batch["K_theta"]
        Lambda = expand_scalar(batch["Lambda"])

        pred_dR = self.physics_loss.d1(pred_p, dim=1, spacing=dR, periodic=False)
        target_dR = self.physics_loss.d1(target_p, dim=1, spacing=dR, periodic=False)
        pred_dTheta = k_theta * self.physics_loss.d1(
            pred_p,
            dim=2,
            spacing=dTheta,
            periodic=True,
            duplicate_endpoint=True,
        )
        target_dTheta = k_theta * self.physics_loss.d1(
            target_p,
            dim=2,
            spacing=dTheta,
            periodic=True,
            duplicate_endpoint=True,
        )
        pred_dZ = Lambda * self.physics_loss.d1(pred_p, dim=3, spacing=dZ, periodic=False)
        target_dZ = Lambda * self.physics_loss.d1(target_p, dim=3, spacing=dZ, periodic=False)

        loss_p_r = self.physics_loss.weighted_mse(pred_dR - target_dR, weight)
        loss_p_theta = self.physics_loss.weighted_mse(pred_dTheta - target_dTheta, weight)
        loss_p_z = self.physics_loss.weighted_mse(pred_dZ - target_dZ, weight)
        total = loss_p_r + loss_p_theta + loss_p_z
        return total, {
            "loss_p": total,
            "loss_p_r": loss_p_r,
            "loss_p_theta": loss_p_theta,
            "loss_p_z": loss_p_z,
        }

    def supervised_loss(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        has_target = batch["has_target"].view(-1, 1, 1, 1)
        if torch.sum(has_target) < 0.5:
            zero = pred["UR"].sum() * 0.0
            return zero, {
                "loss_ur": zero,
                "loss_ut": zero,
                "loss_uz": zero,
                "loss_p": zero,
                "loss_data": zero,
            }

        target = batch["y"]
        target_ur = target[:, 0]
        target_ut = target[:, 1]
        target_uz = target[:, 2]
        target_p = target[:, 3]

        fluid_weight = torch.clamp(batch["phi"], min=0.0, max=1.0) * has_target
        sample_weight = fluid_weight.expand_as(pred["UR"])
        p_grad_weight = self.physics_loss.build_pde_mask(torch.clamp(batch["phi"], min=0.0, max=1.0)) * has_target

        loss_ur = self.physics_loss.weighted_mse(pred["UR"] - target_ur, sample_weight)
        loss_ut = self.physics_loss.weighted_mse(pred["UT"] - target_ut, sample_weight)
        loss_uz = self.physics_loss.weighted_mse(pred["UZ"] - target_uz, sample_weight)
        zero = pred["P"].sum() * 0.0
        if self.pressure_supervision_mode == "gradient":
            loss_p, pressure_log = self.supervised_pressure_gradient_loss(pred["P"], target_p, batch, p_grad_weight)
        elif self.pressure_supervision_mode == "value":
            loss_p = self.physics_loss.weighted_mse(pred["P"] - target_p, fluid_weight)
            pressure_log = {
                "loss_p": loss_p,
                "loss_p_r": zero,
                "loss_p_theta": zero,
                "loss_p_z": zero,
            }
        else:
            loss_p = zero
            pressure_log = {
                "loss_p": zero,
                "loss_p_r": zero,
                "loss_p_theta": zero,
                "loss_p_z": zero,
            }

        total = loss_ur + loss_ut + loss_uz + loss_p
        log = {
            "loss_ur": loss_ur,
            "loss_ut": loss_ut,
            "loss_uz": loss_uz,
            "loss_data": total,
        }
        log.update(pressure_log)
        return total, log

    def current_physics_factor(self, epoch: int) -> float:
        # 先用少量纯数据 warmup，再逐步把物理损失权重拉起来。
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.physics_weight
        ramp = min((epoch - self.warmup_epochs + 1) / self.ramp_epochs, 1.0)
        return self.physics_weight * ramp

    def pure_physics_factor(self) -> float:
        return self.physics_weight

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return (
            "cuda out of memory" in message
            or "cublas_status_alloc_failed" in message
            or "cudnn_status_alloc_failed" in message
        )

    def _empty_cuda_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _raw_batch_size(batch: Mapping[str, torch.Tensor]) -> int:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    @staticmethod
    def _slice_raw_batch(
        batch: Mapping[str, torch.Tensor],
        start: int,
        end: int,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        sliced: dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _estimate_micro_batch_size(self, batch: Mapping[str, torch.Tensor]) -> int:
        batch_size = self._raw_batch_size(batch)
        if batch_size <= 1:
            return batch_size
        if self._runtime_micro_batch_size is not None:
            return min(batch_size, self._runtime_micro_batch_size)
        if not (self.auto_cuda_batching and self.device.type == "cuda"):
            return batch_size

        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        budget = max(1, int(free_bytes * self.cuda_memory_fraction))
        bytes_per_sample = 0
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size:
                bytes_per_sample += int(value[0].numel()) * max(int(value.element_size()), 1)
        grid_n = int(batch["x"].shape[-1]) if "x" in batch and torch.is_tensor(batch["x"]) else 1
        operator_variant = str(self.model_config.get("operator_variant", "")).lower()
        operator_factor = 18 if "3d" in operator_variant else 10
        if grid_n >= 192:
            operator_factor *= 2
        elif grid_n >= 128:
            operator_factor = int(operator_factor * 1.5)
        estimate = max(1, int(budget // max(bytes_per_sample * operator_factor, 1)))
        return min(batch_size, estimate)

    def _compute_micro_batch_log(
        self,
        raw_batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        training: bool,
        loss_scale: float,
        do_backward: bool,
    ) -> dict[str, float]:
        batch = self._to_device(dict(raw_batch))
        batch = self._prepare_runtime_batch(batch)

        with torch.set_grad_enabled(training):
            pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])
            pred = self._apply_kkt_projection(pred, batch)
            loss_data, log_data = self.supervised_loss(pred, batch)
            loss_phys, log_phys = self.physics_loss(pred, batch)
            physics_factor = self.current_physics_factor(epoch)
            loss = self.data_weight * loss_data + physics_factor * loss_phys

            if do_backward:
                (loss * float(loss_scale)).backward()

        merged = {
            "loss_total": loss,
            "physics_factor": torch.tensor(physics_factor, device=loss.device),
            "ibm_C": torch.mean(batch["ibm_C"]),
            "ibm_epsilon": torch.mean(batch["ibm_epsilon"]),
            "has_target": torch.mean(batch["has_target"].to(loss.device)),
            "input_channels": torch.tensor(float(batch["x"].shape[1]), device=loss.device),
            "grid_size": torch.tensor(float(batch["x"].shape[-1]), device=loss.device),
        }
        if "kkt_divergence_before" in pred:
            merged["kkt_div_before"] = torch.mean(pred["kkt_divergence_before"])
            merged["kkt_div_after"] = torch.mean(pred["kkt_divergence_after"])
        merged.update(log_data)
        merged.update(log_phys)

        result = {key: float(value.detach().cpu().item()) for key, value in merged.items()}
        del pred, batch, loss, loss_data, loss_phys
        return result

    def _run_batch_with_auto_split(
        self,
        raw_batch: Mapping[str, torch.Tensor],
        *,
        epoch: int,
        training: bool,
        do_backward: bool,
        step_optimizer: bool,
    ) -> dict[str, float]:
        batch_size = self._raw_batch_size(raw_batch)
        micro_size = max(1, self._estimate_micro_batch_size(raw_batch))

        while True:
            try:
                if do_backward:
                    self.optimizer.zero_grad(set_to_none=True)

                logs: dict[str, float] = {}
                processed = 0
                for start in range(0, batch_size, micro_size):
                    end = min(start + micro_size, batch_size)
                    current = end - start
                    micro_batch = self._slice_raw_batch(raw_batch, start, end, batch_size)
                    micro_log = self._compute_micro_batch_log(
                        micro_batch,
                        epoch=epoch,
                        training=training,
                        loss_scale=current / max(batch_size, 1),
                        do_backward=do_backward,
                    )
                    for key, value in micro_log.items():
                        logs[key] = logs.get(key, 0.0) + value * current
                    processed += current

                if step_optimizer:
                    self.optimizer.step()
                return {key: value / max(processed, 1) for key, value in logs.items()}

            except RuntimeError as exc:
                if not (self.auto_cuda_batching and self._is_cuda_oom(exc) and micro_size > 1):
                    raise
                if do_backward:
                    self.optimizer.zero_grad(set_to_none=True)
                micro_size = max(1, micro_size // 2)
                self._runtime_micro_batch_size = micro_size
                self._empty_cuda_cache()
                print(f"CUDA 显存不足：训练 micro-batch 自动降到 {micro_size}。")

    def run_epoch(self, loader: DataLoader, epoch: int, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
            if self.ibm_mask_controller is not None:
                self.ibm_mask_controller.train()
        else:
            self.model.eval()
            if self.ibm_mask_controller is not None:
                self.ibm_mask_controller.eval()

        logs: dict[str, float] = {}
        count = 0

        for batch in loader:
            raw_count = self._raw_batch_size(batch)
            batch_log = self._run_batch_with_auto_split(
                batch,
                epoch=epoch,
                training=training,
                do_backward=training,
                step_optimizer=training,
            )
            for key, value in batch_log.items():
                logs[key] = logs.get(key, 0.0) + value * raw_count
            count += raw_count

        return {key: value / max(count, 1) for key, value in logs.items()}

    @staticmethod
    def _history_continuity_loss_scale(history: Sequence[Mapping[str, float]]) -> float | None:
        for record in history:
            value = record.get("train_loss_c")
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0.0:
                return value
        return None

    @staticmethod
    def _add_display_scaled_logs(log: dict[str, float], continuity_loss_scale: float) -> None:
        scale = max(float(continuity_loss_scale), 1e-30)
        rmse_scale = float(np.sqrt(scale))
        log["scaled_loss_reference"] = scale
        log["scaled_residual_reference"] = rmse_scale

        for key, value in list(log.items()):
            if not key.startswith("loss_"):
                continue
            try:
                loss_value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(loss_value):
                continue
            log[f"scaled_{key}"] = loss_value / scale

        residual_parts = {
            "c": "loss_c",
            "r": "loss_r",
            "theta": "loss_theta",
            "z": "loss_z",
            "qv": "loss_qv",
        }
        for name, key in residual_parts.items():
            value = log.get(key)
            if value is None:
                continue
            try:
                loss_value = max(float(value), 0.0)
            except (TypeError, ValueError):
                continue
            if np.isfinite(loss_value):
                log[f"scaled_residual_{name}"] = float(np.sqrt(loss_value)) / rmse_scale

        momentum_values = [
            log.get("scaled_loss_r"),
            log.get("scaled_loss_theta"),
            log.get("scaled_loss_z"),
        ]
        if all(value is not None and np.isfinite(float(value)) for value in momentum_values):
            log["scaled_loss_momentum"] = float(np.mean([float(value) for value in momentum_values]))

        residual_momentum_values = [
            log.get("scaled_residual_r"),
            log.get("scaled_residual_theta"),
            log.get("scaled_residual_z"),
        ]
        if all(value is not None and np.isfinite(float(value)) for value in residual_momentum_values):
            log["scaled_residual_momentum"] = float(np.mean([float(value) for value in residual_momentum_values]))

    def fit(
        self,
        epochs: int = 200,
        print_interval: int = 10,
        *,
        start_epoch: int = 0,
        checkpoint_path: str | Path | None = None,
        checkpoint_interval: int = 0,
        history_prefix: Sequence[Mapping[str, float]] | None = None,
        history_plot_path: str | Path | None = None,
        show_history_plot: bool = False,
        history_plot_mode: str = "auto",
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        history_prefix_list = list(history_prefix or [])
        checkpoint_interval = int(max(checkpoint_interval, 0))
        continuity_loss_scale = self._history_continuity_loss_scale(history_prefix_list)

        def save_progress(reason: str) -> None:
            if checkpoint_path is None:
                return
            combined_history = [*history_prefix_list, *history]
            metadata = {
                "checkpoint_reason": reason,
                "completed_new_epochs": len(history),
                "start_epoch": int(start_epoch),
                **dict(checkpoint_metadata or {}),
            }
            self.save_checkpoint(checkpoint_path, history=combined_history, extra_metadata=metadata)
            if history_plot_path is not None and combined_history:
                self.plot_training_history(
                    combined_history,
                    show=show_history_plot,
                    save_path=history_plot_path,
                    history_plot_mode=history_plot_mode,
                )

        try:
            for local_epoch in range(int(epochs)):
                epoch = int(start_epoch) + local_epoch
                train_log = self.run_epoch(self.train_loader, epoch, True)
                val_log = self.run_epoch(self.val_loader, epoch, False)
                if continuity_loss_scale is None:
                    continuity_loss_scale = self._history_continuity_loss_scale(
                        [{"train_loss_c": train_log.get("loss_c", float("nan"))}]
                    )
                if continuity_loss_scale is not None:
                    self._add_display_scaled_logs(train_log, continuity_loss_scale)
                    self._add_display_scaled_logs(val_log, continuity_loss_scale)

                record: dict[str, float] = {}
                for key, value in train_log.items():
                    record[f"train_{key}"] = value
                for key, value in val_log.items():
                    record[f"val_{key}"] = value
                history.append(record)

                if local_epoch == 0 or (local_epoch + 1) % print_interval == 0:
                    print(
                        f"Epoch {epoch + 1:04d} | "
                        f"train_total={train_log['loss_total']:.6e} | "
                        f"train_data={train_log['loss_data']:.6e} | "
                        f"train_phys={train_log['loss_phys']:.6e} | "
                        f"train_qv={train_log['loss_qv']:.6e} | "
                        f"train_scaled_loss_c={train_log.get('scaled_loss_c', float('nan')):.6e} | "
                        f"train_scaled_res_c={train_log.get('scaled_residual_c', float('nan')):.6e} | "
                        f"train_scaled_res_mom={train_log.get('scaled_residual_momentum', float('nan')):.6e} | "
                        f"train_ibm_C={train_log['ibm_C']:.6g} | "
                        f"train_ibm_eps={train_log['ibm_epsilon']:.6g} | "
                        f"train_bc_periodic={train_log['loss_bc_periodic']:.6e} | "
                        f"train_bc_blade={train_log['loss_bc_blade']:.6e} | "
                        f"val_total={val_log['loss_total']:.6e} | "
                        f"val_scaled_loss_c={val_log.get('scaled_loss_c', float('nan')):.6e} | "
                        f"val_scaled_res_c={val_log.get('scaled_residual_c', float('nan')):.6e} | "
                        f"val_scaled_res_mom={val_log.get('scaled_residual_momentum', float('nan')):.6e} | "
                        f"phys_factor={train_log['physics_factor']:.3e}"
                    )

                if checkpoint_interval > 0 and (local_epoch + 1) % checkpoint_interval == 0:
                    save_progress("periodic")
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存当前进度。")
            save_progress("keyboard_interrupt")
            raise
        except BaseException:
            save_progress("exception")
            raise

        save_progress("completed")

        return history


    def configure_training_schedule(
        self,
        *,
        lr: float | None = None,
        data_weight: float | None = None,
        physics_weight: float | None = None,
        warmup_epochs: int | None = None,
        ramp_epochs: int | None = None,
        pressure_supervision_mode: str | None = None,
        pressure_reference_mode: str | None = None,
        pressure_smoothing: float | None = None,
        pressure_highpass_weight: float | None = None,
        pressure_highpass_normalized: bool | None = None,
        physics_discretization: str | None = None,
        rhie_chow_strength: float | None = None,
        momentum_diagonal_floor: float | None = None,
        convection_interpolation: str | None = None,
        fluent_continuity_first5: Sequence[float] | str | None = None,
        fluent_continuity_scale: float | None = None,
    ) -> None:
        # checkpoint 续训时常常只想沿用网络权重，但重新指定混合训练策略。
        # 这个方法不改变网络层数/宽度，只覆盖损失权重和压力处理策略。
        if lr is not None:
            lr = float(lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.trainer_config["lr"] = lr
        if data_weight is not None:
            self.data_weight = float(data_weight)
            self.trainer_config["data_weight"] = self.data_weight
        if physics_weight is not None:
            self.physics_weight = float(physics_weight)
            self.trainer_config["physics_weight"] = self.physics_weight
        if warmup_epochs is not None:
            self.warmup_epochs = int(warmup_epochs)
            self.trainer_config["warmup_epochs"] = self.warmup_epochs
        if ramp_epochs is not None:
            self.ramp_epochs = int(ramp_epochs)
            self.trainer_config["ramp_epochs"] = self.ramp_epochs
        if pressure_supervision_mode is not None:
            mode = str(pressure_supervision_mode).lower()
            if mode not in {"gradient", "value", "none"}:
                raise ValueError("pressure_supervision_mode must be 'gradient', 'value', or 'none'.")
            self.pressure_supervision_mode = mode
            self.model_config["pressure_supervision_mode"] = mode
            self.trainer_config["pressure_supervision_mode"] = mode
        if pressure_reference_mode is not None:
            mode = str(pressure_reference_mode).lower()
            if mode not in {"origin", "none"}:
                raise ValueError("pressure_reference_mode must be 'origin' or 'none'.")
            self.pressure_reference_mode = mode
            self.model.pressure_reference_mode = mode
            self.model_config["pressure_reference_mode"] = mode
            self.trainer_config["pressure_reference_mode"] = mode
        if pressure_smoothing is not None:
            value = float(np.clip(pressure_smoothing, 0.0, 1.0))
            self.model.pressure_smoothing = value
            self.model_config["pressure_smoothing"] = value
            self.trainer_config["pressure_smoothing"] = value
        if pressure_highpass_weight is not None:
            value = float(max(pressure_highpass_weight, 0.0))
            self.physics_loss.pressure_highpass_weight = value
            self.trainer_config["pressure_highpass_weight"] = value
        if pressure_highpass_normalized is not None:
            value = bool(pressure_highpass_normalized)
            self.physics_loss.pressure_highpass_normalized = value
            self.trainer_config["pressure_highpass_normalized"] = value
        if physics_discretization is not None:
            value = str(physics_discretization).lower()
            if value not in {"fvm_rhie_chow", "centered"}:
                raise ValueError("physics_discretization must be 'fvm_rhie_chow' or 'centered'.")
            self.physics_loss.physics_discretization = value
            self.trainer_config["physics_discretization"] = value
        if rhie_chow_strength is not None:
            value = float(max(rhie_chow_strength, 0.0))
            self.physics_loss.rhie_chow_strength = value
            self.trainer_config["rhie_chow_strength"] = value
        if momentum_diagonal_floor is not None:
            value = float(max(momentum_diagonal_floor, 1e-8))
            self.physics_loss.momentum_diagonal_floor = value
            self.trainer_config["momentum_diagonal_floor"] = value
        if convection_interpolation is not None:
            value = _normalize_convection_interpolation(convection_interpolation)
            self.physics_loss.convection_interpolation = value
            self.trainer_config["convection_interpolation"] = value
        if fluent_continuity_first5 is not None or fluent_continuity_scale is not None:
            first5 = (
                _float_sequence_from_any(fluent_continuity_first5)
                if fluent_continuity_first5 is not None
                else self.physics_loss.fluent_continuity_first5
            )
            scale = _resolve_fluent_continuity_scale(fluent_continuity_scale, first5)
            self.physics_loss.fluent_continuity_first5 = first5
            self.physics_loss.fluent_continuity_scale = scale
            self.trainer_config["fluent_continuity_first5"] = list(first5)
            self.trainer_config["fluent_continuity_scale"] = scale

    def smoke_test(self, do_backward: bool = True) -> dict[str, float]:
        # 最小化检查，debug模式
        # 1. DataLoader 是否正常
        # 2. 前向传播是否正常
        # 3. 数据损失 / 物理损失是否都能算
        # 4. backward 是否能打通
        self.model.train()
        if self.ibm_mask_controller is not None:
            self.ibm_mask_controller.train()
        batch = next(iter(self.train_loader))
        result = self._run_batch_with_auto_split(
            batch,
            epoch=0,
            training=do_backward,
            do_backward=do_backward,
            step_optimizer=False,
        )
        return result

    def fit_pure_physics_debug(
        self,
        epochs: int = 20,
        print_interval: int = 1,
        *,
        preview_spans: Sequence[float] = (0.2, 0.5, 0.8),
        post_spans: Sequence[float] = (0.2, 0.5, 0.8),
        show_plots: bool = True,
        save_dir: str | Path | None = None,
        save_checkpoint_path: str | Path | None = None,
        plot_3d: bool = True,
        save_history_plot_path: str | Path | None = None,
        history_plot_mode: str = "auto",
        plot_frequency_energy: bool = True,
    ) -> list[dict[str, float]]:
        # 纯物理调试模式下，先看叶片导入，再训练，再看训练后的场。
        blade_plot_path = None
        post_plot_path = None
        post_3d_path = None
        history_plot_path = None
        frequency_plot_path = None
        frequency_summary_path = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            blade_plot_path = save_dir / "blade_spans.png"
            post_plot_path = save_dir / "post_physical_spans.png"
            post_3d_path = save_dir / "post_3d_streamlines.png"
            history_plot_path = save_dir / "training_loss_log.png"
            frequency_plot_path = save_dir / "frequency_energy_trends.png"
            frequency_summary_path = save_dir / "frequency_energy_summary.json"
            if save_checkpoint_path is None:
                save_checkpoint_path = save_dir / "surrogate_checkpoint.pt"
        if save_history_plot_path is not None:
            history_plot_path = Path(save_history_plot_path)

        print("纯物理调试模式：先展示叶片，再开始训练。")
        self.plot_blade_spans(case_index=0, spans=preview_spans, show=show_plots, save_path=blade_plot_path)
        history = self.fit(epochs=epochs, print_interval=print_interval)
        self.plot_training_history(
            history,
            show=show_plots,
            save_path=history_plot_path,
            history_plot_mode=history_plot_mode,
        )
        if save_checkpoint_path is not None:
            self.save_checkpoint(save_checkpoint_path, history=history)
        self.post_process_case(
            case_index=0,
            spans=post_spans,
            show=show_plots,
            save_path=post_plot_path,
            plot_3d=plot_3d,
            save_path_3d=post_3d_path,
        )
        if plot_frequency_energy and frequency_plot_path is not None:
            self.plot_frequency_energy_trends(
                case_index=0,
                show=show_plots,
                save_path=frequency_plot_path,
                summary_path=frequency_summary_path,
            )
        return history

    @torch.no_grad()
    def predict_case(
        self,
        case: Mapping[str, Any],
        return_physical: bool = False,
    ) -> dict[str, torch.Tensor]:
        bundle = self._predict_case_bundle(case=case)
        return bundle["pred_phy"] if return_physical else bundle["pred_dim"]

    @torch.no_grad()
    def evaluate_cfd_physics_loss(
        self,
        case: Mapping[str, Any],
    ) -> dict[str, float]:
        dataset = BladeFlowDataset(
            [case],
            input_mode=self.train_dataset.input_mode,
            pressure_reference=self.pressure_data_reference,
        )
        sample = dataset[0]
        if float(sample["has_target"].item()) < 0.5:
            raise ValueError("CFD physics loss requires UR/UT/UZ/P target fields in the case.")

        batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
        batch = self._prepare_runtime_batch(batch)
        target = batch["y"]
        pred = {
            "UR": target[:, 0],
            "UT": target[:, 1],
            "UZ": target[:, 2],
            "P": target[:, 3],
        }
        total, logs = self.physics_loss(pred, batch)
        result = {"loss_phys": float(total.detach().cpu().item())}
        result.update({key: float(value.detach().cpu().item()) for key, value in logs.items()})
        return result

    @torch.no_grad()
    def save_checkpoint(
        self,
        path: str | Path,
        *,
        extra_metadata: Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, float]] | None = None,
        save_optimizer: bool = True,
    ) -> None:
        # checkpoint 不仅保存权重，也保存部署时所需的模型结构与训练摘要。
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        payload = {
            "checkpoint_version": 5,
            "model_state_dict": self.model.state_dict(),
            "ibm_mask_controller_state_dict": self.ibm_mask_controller.state_dict() if self.ibm_mask_controller is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "input_mode": self.train_dataset.input_mode,
            "model_config": self.model_config,
            "trainer_config": self.trainer_config,
            "ibm_config": self.ibm_config,
            "train_case_summaries": [case_summary(case) for case in self.train_dataset.cases],
            "val_case_summaries": [case_summary(case) for case in self.val_dataset.cases],
            "history": list(history) if history is not None else None,
            "extra_metadata": dict(extra_metadata or {}),
        }
        try:
            torch.save(payload, str(tmp_path))
            os.replace(str(tmp_path), str(path))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        print(f"模型 checkpoint 已保存到: {path}")

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        payload = load_checkpoint_payload(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        ibm_state = payload.get("ibm_mask_controller_state_dict")
        if self.ibm_mask_controller is not None and ibm_state is not None:
            try:
                self.ibm_mask_controller.load_state_dict(ibm_state)
            except RuntimeError:
                current_state = self.ibm_mask_controller.state_dict()
                compatible_state = {
                    key: value
                    for key, value in ibm_state.items()
                    if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
                }
                skipped = sorted(set(ibm_state) - set(compatible_state))
                self.ibm_mask_controller.load_state_dict(compatible_state, strict=False)
                print(
                    "IBM controller checkpoint partially loaded; "
                    f"skipped incompatible span-profile weights: {skipped}"
                )
        if load_optimizer and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "ibm_config" in payload:
            self.ibm_config = dict(payload["ibm_config"])
        self.checkpoint_metadata = payload
        print(f"模型 checkpoint 已读取: {path}")
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        cases: Sequence[Mapping[str, Any]] | Mapping[str, Any],
        *,
        device: str = "cuda",
        batch_size: int = 1,
        load_optimizer: bool = False,
        pressure_reference_mode: str | None = None,
        pressure_supervision_mode: str | None = None,
        pressure_data_reference: str | None = None,
        pressure_smoothing: float | None = None,
        pressure_highpass_weight: float | None = None,
        pressure_highpass_normalized: bool | None = None,
        physics_discretization: str | None = None,
        rhie_chow_strength: float | None = None,
        momentum_diagonal_floor: float | None = None,
        convection_interpolation: str | None = None,
        fluent_continuity_first5: Sequence[float] | str | None = None,
        fluent_continuity_scale: float | None = None,
    ) -> "SurrogateModeling":
        # 用 checkpoint 重建一个可直接部署的新 trainer。
        path = Path(path)
        payload = load_checkpoint_payload(path, map_location="cpu")
        case_list = [cases] if isinstance(cases, Mapping) else list(cases)

        model_config = dict(payload.get("model_config", {}))
        trainer_config = dict(payload.get("trainer_config", {}))
        input_mode = payload.get("input_mode", trainer_config.get("input_mode", "both"))
        if "operator_variant" not in model_config:
            state_keys = set(payload.get("model_state_dict", {}).keys())
            if any(key.startswith("core.blocks.") and key.endswith(".weights") for key in state_keys):
                model_config["operator_variant"] = "fno"
            else:
                model_config["operator_variant"] = "cfno"

        resolved_pressure_smoothing = (
            float(pressure_smoothing)
            if pressure_smoothing is not None
            else float(model_config.get("pressure_smoothing", trainer_config.get("pressure_smoothing", 0.0)))
        )
        resolved_pressure_reference_mode = str(
            pressure_reference_mode
            if pressure_reference_mode is not None
            else model_config.get("pressure_reference_mode", trainer_config.get("pressure_reference_mode", "origin"))
        )
        resolved_pressure_supervision_mode = str(
            pressure_supervision_mode
            if pressure_supervision_mode is not None
            else trainer_config.get(
                "pressure_supervision_mode",
                model_config.get("pressure_supervision_mode", "gradient"),
            )
        )
        resolved_pressure_data_reference = str(
            pressure_data_reference
            if pressure_data_reference is not None
            else trainer_config.get(
                "pressure_data_reference",
                model_config.get("pressure_data_reference", "training_origin"),
            )
        )
        resolved_pressure_highpass_weight = (
            float(pressure_highpass_weight)
            if pressure_highpass_weight is not None
            else float(trainer_config.get("pressure_highpass_weight", 0.0))
        )
        resolved_pressure_highpass_normalized = (
            bool(pressure_highpass_normalized)
            if pressure_highpass_normalized is not None
            else bool(trainer_config.get("pressure_highpass_normalized", True))
        )
        resolved_physics_discretization = str(
            physics_discretization
            if physics_discretization is not None
            else trainer_config.get("physics_discretization", "fvm_rhie_chow")
        )
        resolved_rhie_chow_strength = (
            float(rhie_chow_strength)
            if rhie_chow_strength is not None
            else float(trainer_config.get("rhie_chow_strength", 0.35))
        )
        resolved_momentum_diagonal_floor = (
            float(momentum_diagonal_floor)
            if momentum_diagonal_floor is not None
            else float(trainer_config.get("momentum_diagonal_floor", 1.0))
        )
        resolved_convection_interpolation = str(
            convection_interpolation
            if convection_interpolation is not None
            else trainer_config.get("convection_interpolation", "central")
        )
        resolved_fluent_continuity_first5 = (
            fluent_continuity_first5
            if fluent_continuity_first5 is not None
            else trainer_config.get("fluent_continuity_first5", None)
        )
        resolved_fluent_continuity_scale = (
            float(fluent_continuity_scale)
            if fluent_continuity_scale is not None
            else trainer_config.get("fluent_continuity_scale", None)
        )

        trainer = cls(
            train_cases=case_list,
            val_cases=case_list,
            input_mode=input_mode,
            batch_size=batch_size,
            lr=float(trainer_config.get("lr", 1e-3)),
            modes=int(model_config.get("modes", 8)),
            width=int(model_config.get("width", 16)),
            depth=int(model_config.get("depth", 4)),
            z_padding=int(model_config.get("z_padding", 8)),
            operator_variant=str(model_config.get("operator_variant", "cfno")),
            high_modes=model_config.get("high_modes", None),
            fourier_feature_bands=tuple(model_config.get("fourier_feature_bands", (1, 2, 4, 8))),
            hf_high_gate_init=float(model_config.get("hf_high_gate_init", trainer_config.get("hf_high_gate_init", -1.0))),
            hf_use_local_highpass=bool(
                model_config.get("hf_use_local_highpass", trainer_config.get("hf_use_local_highpass", True))
            ),
            pressure_smoothing=resolved_pressure_smoothing,
            pressure_reference_mode=resolved_pressure_reference_mode,
            pressure_supervision_mode=resolved_pressure_supervision_mode,
            pressure_data_reference=resolved_pressure_data_reference,
            pressure_highpass_weight=resolved_pressure_highpass_weight,
            pressure_highpass_normalized=resolved_pressure_highpass_normalized,
            physics_discretization=resolved_physics_discretization,
            rhie_chow_strength=resolved_rhie_chow_strength,
            momentum_diagonal_floor=resolved_momentum_diagonal_floor,
            convection_interpolation=resolved_convection_interpolation,
            fluent_continuity_first5=resolved_fluent_continuity_first5,
            fluent_continuity_scale=resolved_fluent_continuity_scale,
            data_weight=float(trainer_config.get("data_weight", 0.0)),
            physics_weight=float(trainer_config.get("physics_weight", 1.0)),
            warmup_epochs=int(trainer_config.get("warmup_epochs", 0)),
            ramp_epochs=int(trainer_config.get("ramp_epochs", 0)),
            use_kkt_projection=bool(trainer_config.get("use_kkt_projection", False)),
            kkt_projection_iters=int(trainer_config.get("kkt_projection_iters", 24)),
            kkt_projection_strength=float(trainer_config.get("kkt_projection_strength", 0.35)),
            learn_ibm_params=bool(trainer_config.get("learn_ibm_params", True)),
            ibm_c_range=tuple(trainer_config.get("ibm_c_range", (0.25, 4.0))),
            ibm_epsilon_range=tuple(trainer_config.get("ibm_epsilon_range", (0.01, 0.08))),
            ibm_hidden=int(trainer_config.get("ibm_hidden", 16)),
            micro_batch_size=trainer_config.get("micro_batch_size", None),
            slice_batch_size=model_config.get("slice_batch_size", trainer_config.get("slice_batch_size", None)),
            auto_cuda_batching=bool(trainer_config.get("auto_cuda_batching", model_config.get("auto_cuda_batching", True))),
            cuda_memory_fraction=float(
                trainer_config.get("cuda_memory_fraction", model_config.get("cuda_memory_fraction", 0.65))
            ),
            use_activation_checkpointing=bool(
                trainer_config.get(
                    "use_activation_checkpointing",
                    model_config.get("use_activation_checkpointing", True),
                )
            ),
            device=device,
        )
        trainer.load_checkpoint(path, load_optimizer=load_optimizer)
        return trainer

    @classmethod
    def deploy_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        case: Mapping[str, Any],
        *,
        device: str = "cuda",
        show: bool = True,
        save_path_2d: str | Path | None = None,
        save_path_3d: str | Path | None = None,
        spans: Sequence[float] = (0.2, 0.5, 0.8),
        plot_3d: bool = True,
        passages_to_plot_3d: int | None = None,
        show_3d: bool | None = None,
    ) -> dict[str, Any]:
        # 这是最直接的部署入口：
        # 读模型 -> 输入新工况与新叶片参数 -> 输出二维 span 图和三维流线图。
        trainer = cls.from_checkpoint(checkpoint_path, case, device=device, batch_size=1, load_optimizer=False)
        return trainer.post_process_case(
            case=case,
            spans=spans,
            show=show,
            save_path=save_path_2d,
            plot_3d=plot_3d,
            save_path_3d=save_path_3d,
            passages_to_plot_3d=passages_to_plot_3d,
            show_3d=show_3d,
        )


def load_cases_from_pt(path: str | Path) -> list[Mapping[str, Any]]:
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "cases" in payload:
        return list(payload["cases"])
    return list(payload)


def load_checkpoint_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> Mapping[str, Any]:
    path = Path(path)
    try:
        payload = torch.load(str(path), map_location=map_location)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read checkpoint. The file may be incomplete or corrupted "
            f"(often caused by an interrupted save): {path}. "
            "Delete or move this file, or choose another checkpoint."
        ) from exc
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise RuntimeError(f"Invalid surrogate checkpoint format: {path}")
    return payload


def is_checkpoint_readable(path: str | Path) -> bool:
    try:
        load_checkpoint_payload(path, map_location="cpu")
    except RuntimeError as exc:
        print(f"Warning: skipping unreadable checkpoint: {path}\n  {exc}")
        return False
    return True


def find_first_blade_params(root: str | Path = ".") -> Path | None:
    # 纯物理调试时，尽量自动找一个 blade_params.json。
    root = Path(root)
    direct = root / "blade_params.json"
    if direct.exists():
        return direct

    matches = list(root.rglob("blade_params.json"))
    if matches:
        return matches[0]
    return None


def _run_token(value: Any) -> str:
    text = str(value).strip().lower()
    allowed = []
    for char in text:
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", "."}:
            allowed.append(char)
        else:
            allowed.append("-")
    token = "".join(allowed).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "run"


def build_formal_run_name(
    *,
    action: str,
    training_mode: str,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    run_suffix: str | None = None,
) -> str:
    model_name = _run_token(model_config.get("operator_variant", "model"))
    n = int(data_config.get("n", 0))
    width = int(model_config.get("width", 0))
    depth = int(model_config.get("depth", 0))
    modes = int(model_config.get("modes", 0))
    high_modes = model_config.get("high_modes", None)
    epochs = int(training_config.get("epochs", 0))
    pressure_supervision = model_config.get("pressure_supervision_mode", None)
    pressure_reference = model_config.get("pressure_reference_mode", None)
    pressure_data_reference = model_config.get("pressure_data_reference", None)
    physics_discretization = training_config.get("physics_discretization", None)
    rhie_chow_strength = training_config.get("rhie_chow_strength", None)
    convection_interpolation = training_config.get("convection_interpolation", None)
    parts = [
        _run_token(action),
        model_name,
        _run_token(training_mode),
        f"n{n:03d}",
        f"m{modes}",
    ]
    if pressure_supervision is not None:
        parts.append("p" + _run_token(pressure_supervision))
    if pressure_reference is not None:
        parts.append("pref" + _run_token(pressure_reference))
    if pressure_data_reference is not None:
        parts.append("pdata" + _run_token(pressure_data_reference))
    if physics_discretization is not None:
        parts.append("disc" + _run_token(physics_discretization))
    if rhie_chow_strength is not None and str(physics_discretization).lower() == "fvm_rhie_chow":
        parts.append("rc" + _run_token(f"{float(rhie_chow_strength):.3g}"))
    if convection_interpolation is not None:
        parts.append("conv" + _run_token(convection_interpolation))
    if high_modes is not None:
        parts.append(f"hm{int(high_modes)}")
    parts.extend([f"w{width}", f"d{depth}", f"ep{epochs}"])
    if run_suffix:
        parts.append(_run_token(run_suffix))
    return "_".join(parts)


def find_latest_checkpoint(root: str | Path = "surrogate_formal") -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    matches = sorted(root.rglob("surrogate_checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    for checkpoint_path in matches:
        if is_checkpoint_readable(checkpoint_path):
            return checkpoint_path
    return None


def resolve_checkpoint_path(path: str | Path | None, *, search_root: str | Path = "surrogate_formal") -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    if str(path).lower() == "latest":
        latest = find_latest_checkpoint(search_root)
        if latest is None:
            raise FileNotFoundError(f"No surrogate_checkpoint.pt found under {search_root}.")
        return latest
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path


def _main_config_path_from_args(argv: Sequence[str]) -> Path | None:
    for index, arg in enumerate(argv[1:]):
        if arg in {"--main-config", "--config"}:
            value_index = index + 2
            if value_index >= len(argv):
                raise ValueError(f"{arg} requires a JSON config path.")
            return Path(argv[value_index])
        if arg.startswith("--main-config="):
            return Path(arg.split("=", 1)[1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    env_path = os.environ.get("SURROGATE_MODELING_CONFIG")
    return Path(env_path) if env_path else None


def load_main_config_override(argv: Sequence[str] | None = None) -> dict[str, Any]:
    path = _main_config_path_from_args(sys.argv if argv is None else argv)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Main config JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Main config JSON must contain an object at the top level.")
    print(f"读取 main 配置覆盖: {path}")
    return payload


def update_mapping_in_place(target: dict[str, Any], override: Mapping[str, Any] | None) -> None:
    if not override:
        return
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            update_mapping_in_place(target[key], value)
        else:
            target[key] = value


def _path_list_from_config(values: Any) -> list[Path]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        values = [values]
    return [Path(str(item)) for item in values if str(item).strip()]


def _path_or_none(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value))


if __name__ == "__main__":
    # ============================================================
    # 0. 正式运行入口
    # ============================================================
    # WORKFLOW_ACTION:
    # train        : 从头训练并保存 checkpoint。
    # resume_train : 读取 checkpoint 后继续训练；CHECKPOINT_TO_LOAD=None 时自动找 surrogate_formal 下最新模型。
    # deploy       : 读取 checkpoint 直接部署/后处理；不会再训练。
    WORKFLOW_ACTION = "resume_train"
    TRAINING_MODE = "mixed"  # mixed / data_only / physics_only
    CHECKPOINT_TO_LOAD = "latest"  #  str|Path|None 或某个 surrogate_checkpoint.pt

    seed_everything(42)
    simulation_folders = [
        Path("../BladeOptimizerLFR/CQ_20260514_115826_SIMULATION"),
        # Path("../BladeOptimizerLFR/CQ_20260519_160122_S01"),
        # 后续多叶片数据集继续追加同构文件夹：UI中需要允许用户添加
    ]

    # ============================================================
    # 1. 物理定义
    # ============================================================
    rpm = -210.0
    physics_config = {
        "mu": 0.006,
        "rho": 10650.0,
        "omega": float(rpm * 2.0 * np.pi / 60.0),
        "qv": 0.025,
        "g": 9.8,
    }

    # ============================================================
    # 2. 网格、模型、训练策略
    # ============================================================
    data_config = {
        "n": 64,
        "theta_sector_index": 0,
        "interpolation_chunk_size": 250_000,
    }
    model_config = {
        "operator_variant": "cfno",
        "input_mode": "both",
        "modes": 8,
        "high_modes": 16,
        "width": 16,
        "depth": 6,
        "z_padding": 8,
        "fourier_feature_bands": (1, 2, 4, 8),
        "hf_high_gate_init": -1.0,
        "hf_use_local_highpass": True,    # 开启高通模式，只在operator_variant = HF_CFNO中有效
        "pressure_smoothing": 0.0,
        # pressure_supervision_mode:
        # value    : 回退到直接用 P 通道做数据监督。
        # gradient : 只用压力梯度做数据监督，适合压力零点不可信的 CSV。
        # none     : 数据损失只监督速度。
        "pressure_supervision_mode": "value",
        # origin 会在网络输出端强制 P(0,0,0)=0；混合训练会在下面自动改成 none。
        "pressure_reference_mode": "origin",
        # value 模式下用 absolute 才能学习并绘制 Fluent 原始 P；gradient 模式可改回 training_origin。
        "pressure_data_reference": "absolute",
    }
    training_config = {
        "epochs": 5000,  # train=总训练轮数；resume_train=本次追加轮数；deploy 会自动置 0。
        "print_interval": 1,
        "checkpoint_interval": 50,  # 长训练时每隔若干 epoch 保存一次，异常/手动中断也会保存。
        "prefer_existing_run_checkpoint": True,  # 同一输出目录已有 checkpoint 时，resume_train 优先接着它跑。
        "batch_size": 1,
        "lr": 1e-3,
        "pressure_highpass_weight": 1e8,
        "pressure_highpass_normalized": True,
        # centered=旧点态差分；fvm_rhie_chow=控制体通量残差 + Rhie-Chow 面速度。
        "physics_discretization": "fvm_rhie_chow",
        "rhie_chow_strength": 0.35,
        "momentum_diagonal_floor": 1.0,
        "convection_interpolation": "upwind2",   # central | upwind (1st order) | upwind2 (2nd order)
        "use_kkt_projection": False,
        "kkt_projection_iters": 24,
        "kkt_projection_strength": 0.35,
        "learn_ibm_params": True,
        "ibm_c_range": (0.3, 3.0),
        "ibm_epsilon_range": (0.001, 0.05),
        "micro_batch_size": None,
        "slice_batch_size": None,
        "auto_cuda_batching": True,
        "cuda_memory_fraction": 0.80,
        "use_activation_checkpointing": True,
        "device": "cuda",
    }
    mode_presets = {
        "mixed": {"data_weight": 1.0, "physics_weight": 0.1, "warmup_epochs": 20, "ramp_epochs": 30},
        "data_only": {"data_weight": 1.0, "physics_weight": 0.0, "warmup_epochs": 0, "ramp_epochs": 0},
        "physics_only": {"data_weight": 0.0, "physics_weight": 1.0, "warmup_epochs": 0, "ramp_epochs": 0},
    }
    main_config_override = load_main_config_override()
    if main_config_override:
        WORKFLOW_ACTION = str(main_config_override.get("workflow_action", WORKFLOW_ACTION))
        TRAINING_MODE = str(main_config_override.get("training_mode", TRAINING_MODE))
        CHECKPOINT_TO_LOAD = main_config_override.get("checkpoint_to_load", CHECKPOINT_TO_LOAD)
        configured_folders = _path_list_from_config(main_config_override.get("simulation_folders"))
        if configured_folders:
            simulation_folders = configured_folders
        cfd_csv_files = _path_list_from_config(
            main_config_override.get("cfd_csv_files", main_config_override.get("cfd_csv", None))
        )
        if "rpm" in main_config_override:
            rpm = float(main_config_override["rpm"])
            physics_config["omega"] = float(rpm * 2.0 * np.pi / 60.0)
        update_mapping_in_place(physics_config, main_config_override.get("physics_config"))
        update_mapping_in_place(data_config, main_config_override.get("data_config"))
        update_mapping_in_place(model_config, main_config_override.get("model_config"))
        update_mapping_in_place(training_config, main_config_override.get("training_config"))
    else:
        cfd_csv_files = []
    if WORKFLOW_ACTION not in {"train", "resume_train", "deploy"}:
        raise ValueError("WORKFLOW_ACTION must be 'train', 'resume_train', or 'deploy'.")
    if TRAINING_MODE not in mode_presets:
        raise ValueError(f"Unknown TRAINING_MODE={TRAINING_MODE!r}; choose from {sorted(mode_presets)}.")
    resolved_simulation_folders: list[Path] = []
    for folder in simulation_folders:
        blade_params = find_first_blade_params(folder)
        if blade_params is None:
            resolved_simulation_folders.append(Path(folder))
        else:
            resolved_simulation_folders.append(blade_params.parent)
            if blade_params.parent != Path(folder):
                print(f"在检查文件夹中找到 blade_params.json: {blade_params}")
    simulation_folders = resolved_simulation_folders
    training_config.update(mode_presets[TRAINING_MODE])
    if (
        model_config["pressure_supervision_mode"] == "value"
        and model_config["pressure_data_reference"] == "absolute"
    ):
        # 直接学习 Fluent 原始 P 时，不能再从模型输出端扣掉一个固定压力零点。
        model_config["pressure_reference_mode"] = "none"
    if TRAINING_MODE == "mixed":
        # 混合训练时，P 应由数据项决定；这里关闭会额外“规定压力形态”的物理/模型约束。
        training_config["pressure_highpass_weight"] = 0.0
        model_config["pressure_smoothing"] = 0.0
        model_config["pressure_reference_mode"] = "none"
    if WORKFLOW_ACTION == "deploy":
        training_config["epochs"] = 0

    # ============================================================
    # 3. 正式输出目录
    # ============================================================
    output_root = Path(main_config_override.get("output_root", "surrogate_formal"))
    checkpoint_request = CHECKPOINT_TO_LOAD
    if WORKFLOW_ACTION in {"resume_train", "deploy"} and checkpoint_request is None:
        checkpoint_request = "latest"
    checkpoint_to_load = resolve_checkpoint_path(checkpoint_request, search_root=output_root)

    naming_model_config = dict(model_config)
    if checkpoint_to_load is not None:
        try:
            checkpoint_payload_for_name = load_checkpoint_payload(checkpoint_to_load, map_location="cpu")
        except RuntimeError as exc:
            if WORKFLOW_ACTION == "train":
                print(f"Warning: ignoring unreadable checkpoint for fresh training: {checkpoint_to_load}\n  {exc}")
                checkpoint_to_load = None
                checkpoint_payload_for_name = None
            else:
                raise
        if checkpoint_payload_for_name is not None:
            naming_model_config.update(dict(checkpoint_payload_for_name.get("model_config", {})))
        if WORKFLOW_ACTION == "resume_train":
            # 架构信息可沿用 checkpoint；续训显式选择的压力策略必须以主程序配置为准。
            for key in (
                "pressure_supervision_mode",
                "pressure_reference_mode",
                "pressure_data_reference",
                "pressure_smoothing",
            ):
                naming_model_config[key] = model_config[key]
    run_suffix = simulation_folders[0].name if len(simulation_folders) == 1 else f"{len(simulation_folders)}cases"
    run_name = build_formal_run_name(
        action=WORKFLOW_ACTION,
        training_mode=TRAINING_MODE,
        model_config=naming_model_config,
        data_config=data_config,
        training_config=training_config,
        run_suffix=run_suffix,
    )
    save_dir = output_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "surrogate_checkpoint.pt"
    if (
        WORKFLOW_ACTION == "resume_train"
        and bool(training_config.get("prefer_existing_run_checkpoint", True))
        and checkpoint_path.exists()
    ):
        if is_checkpoint_readable(checkpoint_path):
            checkpoint_to_load = checkpoint_path

    print(f"\n========== Surrogate formal workflow: {WORKFLOW_ACTION} / {TRAINING_MODE} ==========")
    print(f"输出目录: {save_dir}")
    if checkpoint_to_load is not None:
        print(f"读取 checkpoint: {checkpoint_to_load}")
    print(
        "物理定义: "
        f"rpm={rpm:.6g}, omega={physics_config['omega']:.6g}, "
        f"mu={physics_config['mu']:.6g}, rho={physics_config['rho']:.6g}, "
        f"qv={physics_config['qv']:.6g}, g={physics_config['g']:.6g}"
    )

    # ============================================================
    # 4. 后处理/可视化定义
    # ============================================================
    # show_pyvista_window=True 会弹出 PyVista 交互窗口；Matplotlib 图仍默认只保存不弹窗。
    post_config = {
        "spans": (0.4, 0.6),
        "show_matplotlib": False,
        "show_pyvista_window": True,
        "plot_3d": True,
        "history_plot_mode": "all",  # all saves separate raw / scaled_loss / scaled_residual plots.
        "passages_to_plot_3d": None,  # None=先算单流道，再复制到全部 n_blade 个周期。
        "cfd_pressure_reference": "absolute",
        "interpolation_chunk_size": data_config["interpolation_chunk_size"],
    }
    update_mapping_in_place(post_config, main_config_override.get("post_config"))

    # ============================================================
    # 5. 构造训练/部署样本
    # ============================================================
    if TRAINING_MODE == "physics_only" and not cfd_csv_files:
        blade_param_files = [folder / "blade_params.json" for folder in simulation_folders]
        train_cases = make_pure_physics_debug_cases(
            blade_params=blade_param_files,
            n=data_config["n"],
            **physics_config,
        )
    else:
        train_cases = []
        for index, folder in enumerate(simulation_folders):
            csv_path = None
            if cfd_csv_files:
                csv_path = cfd_csv_files[index] if index < len(cfd_csv_files) else cfd_csv_files[-1]
            train_cases.append(
                make_supervised_simulation_case(
                    folder,
                    n=data_config["n"],
                    theta_sector_index=data_config["theta_sector_index"],
                    interpolation_chunk_size=data_config["interpolation_chunk_size"],
                    csv_path=csv_path,
                    **physics_config,
                )
            )
    val_cases = train_cases
    has_cfd_targets = all(_pick(train_cases[0], name) is not None for name in ["UR", "UT", "UZ", "P"])

    first_config = FlowCaseConfig.from_mapping(train_cases[0])
    run_summary = {
        "workflow_action": WORKFLOW_ACTION,
        "training_mode": TRAINING_MODE,
        "simulation_folders": [str(folder) for folder in simulation_folders],
        "cfd_csv_files": [str(path) for path in cfd_csv_files],
        "checkpoint_to_load": str(checkpoint_to_load) if checkpoint_to_load is not None else None,
        "checkpoint_to_save": str(checkpoint_path) if WORKFLOW_ACTION != "deploy" else None,
        "physics_config": {"rpm": rpm, **physics_config, "g_star": first_config.g_star, "P0": first_config.P0},
        "data_config": data_config,
        "model_config": naming_model_config,
        "training_config": training_config,
        "post_config": post_config,
        "case_summaries": [case_summary(case) for case in train_cases],
    }
    summary_path = save_dir / "run_config_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"运行配置摘要已保存到: {summary_path}")

    # ============================================================
    # 6. 建立/读取模型
    # ============================================================
    loaded_history: list[Mapping[str, float]] = []
    if checkpoint_to_load is not None:
        trainer = SurrogateModeling.from_checkpoint(
            checkpoint_to_load,
            train_cases,
            device=training_config["device"],
            batch_size=training_config["batch_size"],
            load_optimizer=WORKFLOW_ACTION == "resume_train",
            pressure_reference_mode=model_config["pressure_reference_mode"] if WORKFLOW_ACTION == "resume_train" else None,
            pressure_supervision_mode=(
                model_config["pressure_supervision_mode"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            pressure_data_reference=model_config["pressure_data_reference"] if WORKFLOW_ACTION == "resume_train" else None,
            pressure_smoothing=model_config["pressure_smoothing"] if WORKFLOW_ACTION == "resume_train" else None,
            pressure_highpass_weight=(
                training_config["pressure_highpass_weight"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            pressure_highpass_normalized=(
                training_config["pressure_highpass_normalized"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            physics_discretization=(
                training_config["physics_discretization"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            rhie_chow_strength=training_config["rhie_chow_strength"] if WORKFLOW_ACTION == "resume_train" else None,
            momentum_diagonal_floor=(
                training_config["momentum_diagonal_floor"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            convection_interpolation=(
                training_config["convection_interpolation"] if WORKFLOW_ACTION == "resume_train" else None
            ),
            fluent_continuity_first5=training_config.get("fluent_continuity_first5"),
            fluent_continuity_scale=training_config.get("fluent_continuity_scale"),
        )
        loaded_history = list((trainer.checkpoint_metadata or {}).get("history") or [])
        if WORKFLOW_ACTION == "resume_train":
            trainer.configure_training_schedule(
                lr=training_config["lr"],
                data_weight=training_config["data_weight"],
                physics_weight=training_config["physics_weight"],
                warmup_epochs=training_config["warmup_epochs"],
                ramp_epochs=training_config["ramp_epochs"],
                pressure_supervision_mode=model_config["pressure_supervision_mode"],
                pressure_reference_mode=model_config["pressure_reference_mode"],
                pressure_smoothing=model_config["pressure_smoothing"],
                pressure_highpass_weight=training_config["pressure_highpass_weight"],
                pressure_highpass_normalized=training_config["pressure_highpass_normalized"],
                physics_discretization=training_config["physics_discretization"],
                rhie_chow_strength=training_config["rhie_chow_strength"],
                momentum_diagonal_floor=training_config["momentum_diagonal_floor"],
                convection_interpolation=training_config["convection_interpolation"],
                fluent_continuity_first5=training_config.get("fluent_continuity_first5"),
                fluent_continuity_scale=training_config.get("fluent_continuity_scale"),
            )
    else:
        trainer = SurrogateModeling(
            train_cases=train_cases,
            val_cases=val_cases,
            input_mode=model_config["input_mode"],
            batch_size=training_config["batch_size"],
            lr=training_config["lr"],
            modes=model_config["modes"],
            high_modes=model_config["high_modes"],
            width=model_config["width"],
            depth=model_config["depth"],
            z_padding=model_config["z_padding"],
            operator_variant=model_config["operator_variant"],
            fourier_feature_bands=model_config["fourier_feature_bands"],
            hf_high_gate_init=model_config["hf_high_gate_init"],
            hf_use_local_highpass=model_config["hf_use_local_highpass"],
            pressure_smoothing=model_config["pressure_smoothing"],
            pressure_reference_mode=model_config["pressure_reference_mode"],
            pressure_supervision_mode=model_config["pressure_supervision_mode"],
            pressure_data_reference=model_config["pressure_data_reference"],
            pressure_highpass_weight=training_config["pressure_highpass_weight"],
            pressure_highpass_normalized=training_config["pressure_highpass_normalized"],
            physics_discretization=training_config["physics_discretization"],
            rhie_chow_strength=training_config["rhie_chow_strength"],
            momentum_diagonal_floor=training_config["momentum_diagonal_floor"],
            convection_interpolation=training_config["convection_interpolation"],
            fluent_continuity_first5=training_config.get("fluent_continuity_first5"),
            fluent_continuity_scale=training_config.get("fluent_continuity_scale"),
            data_weight=training_config["data_weight"],
            physics_weight=training_config["physics_weight"],
            warmup_epochs=training_config["warmup_epochs"],
            ramp_epochs=training_config["ramp_epochs"],
            use_kkt_projection=training_config["use_kkt_projection"],
            kkt_projection_iters=training_config["kkt_projection_iters"],
            kkt_projection_strength=training_config["kkt_projection_strength"],
            learn_ibm_params=training_config["learn_ibm_params"],
            ibm_c_range=training_config["ibm_c_range"],
            ibm_epsilon_range=training_config["ibm_epsilon_range"],
            micro_batch_size=training_config["micro_batch_size"],
            slice_batch_size=training_config["slice_batch_size"],
            auto_cuda_batching=training_config["auto_cuda_batching"],
            cuda_memory_fraction=training_config["cuda_memory_fraction"],
            use_activation_checkpointing=training_config["use_activation_checkpointing"],
            device=training_config["device"],
        )

    # ============================================================
    # 7. 训练或续训
    # ============================================================
    trainer.plot_blade_spans(
        case_index=0,
        spans=post_config["spans"],
        show=post_config["show_matplotlib"],
        save_path=save_dir / "blade_spans.png",
    )

    history: list[Mapping[str, float]] = list(loaded_history)
    if WORKFLOW_ACTION in {"train", "resume_train"}:
        smoke = trainer.smoke_test(do_backward=True)
        print("Smoke test:", smoke)
        new_history = trainer.fit(
            epochs=training_config["epochs"],
            print_interval=training_config["print_interval"],
            start_epoch=len(loaded_history),
            checkpoint_path=checkpoint_path,
            checkpoint_interval=training_config["checkpoint_interval"],
            history_prefix=loaded_history,
            history_plot_path=save_dir / "training_loss_log.png",
            show_history_plot=post_config["show_matplotlib"],
            history_plot_mode=post_config["history_plot_mode"],
            checkpoint_metadata={"run_summary_path": str(summary_path), "workflow_action": WORKFLOW_ACTION},
        )
        history = [*loaded_history, *new_history]
        trainer.plot_training_history(
            history,
            show=post_config["show_matplotlib"],
            save_path=save_dir / "training_loss_log.png",
            history_plot_mode=post_config["history_plot_mode"],
        )
        trainer.save_checkpoint(
            checkpoint_path,
            history=history,
            extra_metadata={"run_summary_path": str(summary_path), "workflow_action": WORKFLOW_ACTION},
        )
    else:
        print("部署模式：已跳过训练，仅执行后处理。")

    # ============================================================
    # 8. CFD、预测和误差诊断
    # ============================================================
    if has_cfd_targets:
        cfd_physics_loss = trainer.evaluate_cfd_physics_loss(train_cases[0])
        cfd_physics_loss_path = save_dir / "cfd_physics_loss_summary.json"
        cfd_physics_loss_path.write_text(
            json.dumps(cfd_physics_loss, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"CFD 物理 loss 诊断已保存到: {cfd_physics_loss_path}")

        trainer.plot_cfd_spans(
            case_index=0,
            spans=post_config["spans"],
            show=post_config["show_matplotlib"],
            save_path=save_dir / "cfd_physical_spans.png",
            pressure_reference=post_config["cfd_pressure_reference"],
            interpolation_chunk_size=post_config["interpolation_chunk_size"],
        )
        trainer.plot_local_physics_residual_spans(
            case_index=0,
            spans=post_config["spans"],
            source="cfd",
            show=post_config["show_matplotlib"],
            save_path=save_dir / "cfd_local_residual_spans.png",
            summary_path=save_dir / "cfd_local_residual_summary.json",
        )
        compare_diagnostics = trainer.compare_cfd_prediction_spans(
            case_index=0,
            spans=post_config["spans"],
            show=post_config["show_matplotlib"],
            save_path=save_dir / "cfd_vs_nn_error_spans.png",
            summary_path=save_dir / "cfd_vs_nn_error_summary.json",
            pressure_reference=post_config["cfd_pressure_reference"],
            interpolation_chunk_size=post_config["interpolation_chunk_size"],
        )
        compare_diagnostics["cfd_physics_loss"] = cfd_physics_loss
        (save_dir / "cfd_vs_nn_error_summary.json").write_text(
            json.dumps(compare_diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        print("当前 case 没有 CFD 标签，已跳过 CFD span 和误差对比图。")

    # ============================================================
    # 9. 神经网络预测后处理
    # ============================================================
    trainer.plot_local_physics_residual_spans(
        case_index=0,
        spans=post_config["spans"],
        source="nn",
        show=post_config["show_matplotlib"],
        save_path=save_dir / "nn_local_residual_spans.png",
        summary_path=save_dir / "nn_local_residual_summary.json",
    )
    trainer.post_process_case(
        case_index=0,
        spans=post_config["spans"],
        show=post_config["show_matplotlib"],
        save_path=save_dir / "nn_physical_spans.png",
        plot_3d=post_config["plot_3d"],
        save_path_3d=save_dir / "nn_3d_streamlines_fullwheel.png",
        passages_to_plot_3d=post_config["passages_to_plot_3d"],
        show_3d=post_config["show_pyvista_window"],
    )
    trainer.plot_frequency_energy_trends(
        case_index=0,
        show=post_config["show_matplotlib"],
        save_path=save_dir / "frequency_energy_trends.png",
        summary_path=save_dir / "frequency_energy_summary.json",
    )

    # ============================================================
    # 10. 可选：细网格部署
    # ============================================================
    run_fine_grid_deploy = bool(main_config_override.get("run_fine_grid_deploy", False))
    if run_fine_grid_deploy:
        fine_case = make_pure_physics_debug_case(
            blade_params=simulation_folders[0] / "blade_params.json",
            n=256,
            **physics_config,
        )
        deploy_checkpoint = checkpoint_path if WORKFLOW_ACTION != "deploy" else checkpoint_to_load
        if deploy_checkpoint is None:
            raise RuntimeError("Fine-grid deployment requires a checkpoint.")
        SurrogateModeling.deploy_from_checkpoint(
            deploy_checkpoint,
            fine_case,
            show=post_config["show_matplotlib"],
            show_3d=post_config["show_pyvista_window"],
            save_path_2d=save_dir / "fine_grid_nn_spans.png",
            save_path_3d=save_dir / "fine_grid_3d_streamlines_fullwheel.png",
            spans=post_config["spans"],
            plot_3d=True,
            passages_to_plot_3d=None,
        )
