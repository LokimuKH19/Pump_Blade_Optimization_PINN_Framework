from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import NeuralOperators
from BladeImport import PassageGeometry, build_blade_boundary
from SurrogateModelingUtils import (
    case_summary,
    d1_periodic_with_overlap,
    d2_periodic_with_overlap,
    expand_scalar,
    field_stats,
    interpolate_field_periodic,
    line_quadrature_weight,
    make_pyvista_blade_surface_meshes,
    make_pyvista_passage_grid,
    neighbor_minus,
    neighbor_plus,
)


# 这个文件负责把“叶片场 -> 流场”的代理建模流程接起来。
# 目前网络主体仍然调用 NeuralOperators.py 里的标准 2D FNO。
# 因此这里采用“沿 R 方向逐层共享权重”的方式：
# 每个半径层是一个 Theta-Z 平面，所有半径层共用同一套 2D FNO 权重。
#
# 输入方面默认同时使用 blade_mask 和 phi：
# 1. blade_mask 提供清晰的固体拓扑；
# 2. phi 提供适合硬约束和物理损失加权的平滑浸没边界信息。


def seed_everything(seed: int) -> None:
    # 统一随机种子，便于复现实验和排查问题。
    NeuralOperators.seed_everything(seed)


def _pick(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    # 在多个候选键中取第一个存在的值。
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


@dataclass
class FlowCaseConfig:
    # 单个样本或单个工况需要的全部几何、物性与尺度参数。
    n: int
    rh: float
    rs: float
    h: float
    mu: float
    rho: float
    omega: float
    qv: float
    n_blade: int
    z0: float = 0.0
    g: float = 9.8
    g_star: float = 0.0
    ibm_C: float = 1.0
    ibm_epsilon: float = 0.025
    absolute_frame: bool = True
    use_absolute_omega_scale: bool = True

    @classmethod
    def from_mapping(cls, case: Mapping[str, Any]) -> "FlowCaseConfig":
        return cls(
            n=int(_pick(case, "n")),
            rh=float(_pick(case, "rh")),
            rs=float(_pick(case, "rs")),
            h=float(_pick(case, "h")),
            mu=float(_pick(case, "mu")),
            rho=float(_pick(case, "rho")),
            omega=float(_pick(case, "omega")),
            qv=float(_pick(case, "qv")),
            n_blade=int(_pick(case, "n_blade", "n_blades")),
            z0=float(_pick(case, "z0", default=0.0)),
            g=float(_pick(case, "g", default=9.8)),
            g_star=float(_pick(case, "g_star", default=0.0)),
            ibm_C=float(_pick(case, "ibm_C", default=1.0)),
            ibm_epsilon=float(_pick(case, "ibm_epsilon", default=0.025)),
            absolute_frame=bool(_pick(case, "absolute_frame", default=True)),
            use_absolute_omega_scale=bool(_pick(case, "use_absolute_omega_scale", default=True)),
        )

    @property
    def delta_r(self) -> float:
        return self.rs - self.rh

    @property
    def theta0(self) -> float:
        return 2.0 * np.pi / self.n_blade

    @property
    def dR(self) -> float:
        return 1.0 / (self.n - 1)

    @property
    def dTheta(self) -> float:
        # Theta 网格与 BladeImport 保持一致，首尾点重合，因此步长仍写作 1 / (n - 1)。
        return 1.0 / (self.n - 1)

    @property
    def dZ(self) -> float:
        return 1.0 / (self.n - 1)

    @property
    def u_omega(self) -> float:
        # 周向尺度默认取叶顶圆周速度的绝对值。
        tip_speed = self.rs * self.omega
        if self.use_absolute_omega_scale:
            return max(abs(tip_speed), 1e-12)
        return tip_speed if abs(tip_speed) > 1e-12 else 1e-12

    @property
    def u_zo(self) -> float:
        # 轴向尺度取整圈流道上的平均轴向速度。
        return self.qv / (np.pi * (self.rs ** 2 - self.rh ** 2))

    @property
    def P0(self) -> float:
        return self.rho * (self.u_omega ** 2 / 2.0 + self.u_zo ** 2 / 2.0 + self.g * self.h)

    @property
    def Re_omega(self) -> float:
        return self.u_omega * self.delta_r / (self.mu / self.rho)

    @property
    def Eu_omega(self) -> float:
        return self.P0 / (self.rho * self.u_omega ** 2)

    @property
    def Lambda(self) -> float:
        return self.delta_r / self.h

    @property
    def Ku(self) -> float:
        return self.u_zo / self.u_omega

    @property
    def delta(self) -> float:
        return self.delta_r / self.rs

    @property
    def sgn_omega(self) -> float:
        return 1.0 if self.omega >= 0 else -1.0

    @property
    def qv_passage(self) -> float:
        # 总流量 qv 是整圈的，单个流道只承担 1 / n_blade。
        return self.qv / self.n_blade

    @property
    def qv_hat(self) -> float:
        # 对应单流道的无量纲流量：
        # q_hat = q_passage / (u_zo * delta_r^2 * theta0)
        return self.qv_passage / (max(abs(self.u_zo), 1e-12) * self.delta_r ** 2 * self.theta0)

    def make_passage_geometry(self) -> PassageGeometry:
        return PassageGeometry(
            hub_radius=self.rh,
            shroud_radius=self.rs,
            passage_height=self.h,
            blade_count=self.n_blade,
            grid_size=self.n,
            passage_z0=self.z0,
        )


def build_geometry_tensors(config: FlowCaseConfig) -> dict[str, torch.Tensor]:
    # 构造和网格一一对应的几何辅助场。
    # Theta 方向保留首尾重合点，这样与 BladeImport 输出的 mask / phi 完全对齐。
    r = torch.linspace(0.0, 1.0, config.n, dtype=torch.float32)
    theta = torch.linspace(0.0, 1.0, config.n, dtype=torch.float32)
    z = torch.linspace(0.0, 1.0, config.n, dtype=torch.float32)
    rr, tt, zz = torch.meshgrid(r, theta, z, indexing="ij")

    r_hat = rr + config.rh / config.delta_r
    k_theta = 1.0 / (r_hat * config.theta0)
    solid_ut = config.delta * r_hat
    theta_phase = 2.0 * np.pi * tt
    theta_sin = torch.sin(theta_phase)
    theta_cos = torch.cos(theta_phase)

    return {
        "R": rr,
        "Theta": tt,
        "Z": zz,
        "r_hat": r_hat,
        "K_theta": k_theta,
        "solid_ut": solid_ut,
        "Theta_sin": theta_sin,
        "Theta_cos": theta_cos,
    }


def build_phi_from_signed_distance(
    signed_distance: torch.Tensor,
    ibm_C: float | torch.Tensor,
    ibm_epsilon: float | torch.Tensor,
) -> torch.Tensor:
    # 这里和 DataGenerator 中的浸没边界写法保持一致：
    # phi=1 近似纯流体，phi=0 近似纯固体，中间是平滑过渡层。
    if torch.is_tensor(ibm_C):
        c_value = ibm_C.to(device=signed_distance.device, dtype=signed_distance.dtype)
    else:
        c_value = torch.tensor(float(ibm_C), device=signed_distance.device, dtype=signed_distance.dtype)

    if torch.is_tensor(ibm_epsilon):
        epsilon_value = ibm_epsilon.to(device=signed_distance.device, dtype=signed_distance.dtype)
    else:
        epsilon_value = torch.tensor(float(ibm_epsilon), device=signed_distance.device, dtype=signed_distance.dtype)

    epsilon_value = torch.clamp(epsilon_value, min=1e-8)
    return 1.0 - torch.exp(-c_value * signed_distance ** 2 / (epsilon_value ** 2))


def hard_project_theta_periodic(field: torch.Tensor, theta_dim: int) -> torch.Tensor:
    # 当前 Theta 网格包含首尾两个重合点。
    # 这里把首尾两个切片直接投影到同一个拼缝值上，保证数值上严格周期。
    left_index = [slice(None)] * field.ndim
    right_index = [slice(None)] * field.ndim
    left_index[theta_dim] = 0
    right_index[theta_dim] = -1

    left = field[tuple(left_index)]
    right = field[tuple(right_index)]
    seam = 0.5 * (left + right)

    out = field.clone()
    out[tuple(left_index)] = seam
    out[tuple(right_index)] = seam
    return out


def span_to_index(span: float, n: int) -> int:
    span = float(np.clip(span, 0.0, 1.0))
    return int(round(span * (n - 1)))


def normalize_target_fields(
    case: Mapping[str, Any],
    config: FlowCaseConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 监督标签采用 [UR, UT, UZ, P] 四通道。
    # 没有标签时允许走“纯物理调试模式”，此时返回零张量并置 has_target=0。
    ur_raw = _pick(case, "UR", "Ur", "ur")
    ut_raw = _pick(case, "UT", "Ut", "ut")
    uz_raw = _pick(case, "UZ", "Uz", "uz")
    p_raw = _pick(case, "P", "p")

    has_any = any(item is not None for item in [ur_raw, ut_raw, uz_raw, p_raw])
    has_all = all(item is not None for item in [ur_raw, ut_raw, uz_raw, p_raw])

    if has_any and not has_all:
        raise ValueError("UR, UT, UZ, P must be either all provided or all omitted.")

    if not has_any:
        zero = torch.zeros((config.n, config.n, config.n), dtype=torch.float32)
        target = torch.stack([zero, zero, zero, zero], dim=0)
        return target, torch.tensor(0.0, dtype=torch.float32)

    ur = _to_tensor(ur_raw)
    ut = _to_tensor(ut_raw)
    uz = _to_tensor(uz_raw)
    p = _to_tensor(p_raw)

    fields_are_dimensionless = bool(_pick(case, "fields_are_dimensionless", default=True))
    if not fields_are_dimensionless:
        ur = ur / config.u_omega
        ut = ut / config.u_omega
        uz = uz / config.u_zo
        p = p / config.P0

    # 压力统一减去参考点，和网络前向中的压力参考保持一致。
    p = p - p[0, 0, 0]
    target = torch.stack([ur, ut, uz, p], dim=0).to(torch.float32)
    return target, torch.tensor(1.0, dtype=torch.float32)


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
    ):
        self.cases = list(cases)
        self.input_mode = input_mode
        self.samples = [self._build_sample(case) for case in self.cases]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def _build_sample(self, case: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        config = FlowCaseConfig.from_mapping(case)
        geometry = build_geometry_tensors(config)
        blade_mask, phi, signed_distance = self._build_blade_channels(case, config)
        target, has_target = normalize_target_fields(case, config)
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
            "ibm_C": torch.tensor(config.ibm_C, dtype=torch.float32),
            "ibm_epsilon": torch.tensor(config.ibm_epsilon, dtype=torch.float32),
            "absolute_frame": torch.tensor(1.0 if config.absolute_frame else 0.0, dtype=torch.float32),
        }

    def _build_blade_channels(
        self,
        case: Mapping[str, Any],
        config: FlowCaseConfig,
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
            phi = build_phi_from_signed_distance(signed_distance, config.ibm_C, config.ibm_epsilon)

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
    ):
        super().__init__()
        self.z_padding = int(max(z_padding, 0))
        # todo 在这里替换网络类型
        self.core = NeuralOperators.CFNO2d_small(
            modes=modes,
            cheb_modes=(modes, modes),
            width=width,
            depth=depth,
            input_features=input_channels,
            output_features=4,
        )

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
        x_slice = x.permute(0, 2, 1, 3, 4).reshape(batch_size * n_r, in_channels, n_theta, n_z)

        if self.z_padding > 0:
            x_slice = F.pad(x_slice, (self.z_padding, self.z_padding, 0, 0), mode="replicate")

        raw = self.core(x_slice)

        if self.z_padding > 0:
            raw = raw[..., self.z_padding:-self.z_padding]

        raw = raw.reshape(batch_size, n_r, 4, n_theta, n_z).permute(0, 2, 1, 3, 4)

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

        # 压力参考点直接固定在 (0,0,0)。
        p_ref = p[:, 0, 0, 0].view(batch_size, 1, 1, 1)
        p = p - p_ref

        return {
            "UR": ur,
            "UT": ut,
            "UZ": uz,
            "P": p,
        }


class AdaptiveIBMMaskController(nn.Module):
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
            nn.Linear(8, self.hidden),
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
        blade_fraction = torch.mean(batch["blade_mask"], dim=(1, 2, 3))
        signed_distance_abs = torch.abs(batch["signed_distance"])
        signed_distance_mean = torch.mean(signed_distance_abs, dim=(1, 2, 3))
        signed_distance_std = torch.std(signed_distance_abs, dim=(1, 2, 3), unbiased=False)

        qv_hat = batch["qv_hat"].view(-1)
        re_log = torch.log(torch.clamp(batch["Re_omega"].view(-1), min=1e-12))
        eu_log = torch.log(torch.clamp(batch["Eu_omega"].view(-1), min=1e-12))
        lambda_value = batch["Lambda"].view(-1)
        delta_value = batch["delta"].view(-1)

        return torch.stack(
            [
                blade_fraction,
                signed_distance_mean,
                signed_distance_std,
                qv_hat,
                re_log,
                eu_log,
                lambda_value,
                delta_value,
            ],
            dim=1,
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(batch)
        offset = 0.5 * torch.tanh(self.mlp(features))
        raw_value = self.base_logit.view(1, 2) + offset

        ibm_c = self._map_to_range(raw_value[:, 0], self.c_range[0], self.c_range[1]).view(-1, 1, 1, 1)
        ibm_epsilon = self._map_to_range(raw_value[:, 1], self.epsilon_range[0], self.epsilon_range[1]).view(-1, 1, 1, 1)
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
    def __init__(self):
        super().__init__()

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

        # 出口流量损失使用无量纲形式，这样和其余 PDE 残差的量纲更一致。
        q_hat_pred = self.outlet_flow_rate_hat(uz, batch)
        q_hat_target = batch["qv_hat"]
        loss_qv = torch.mean((q_hat_pred - q_hat_target) ** 2)

        # 下面两个量只是用于检查边界条件是否真的被硬约束住。
        loss_bc_periodic = (
            self.theta_periodic_error(ur)
            + self.theta_periodic_error(ut)
            + self.theta_periodic_error(uz)
            + self.theta_periodic_error(p)
        )
        loss_bc_blade = self.blade_noslip_error(pred, batch)

        total = loss_c + loss_r + loss_theta + loss_z + loss_qv
        return total, {
            "loss_c": loss_c,
            "loss_r": loss_r,
            "loss_theta": loss_theta,
            "loss_z": loss_z,
            "loss_qv": loss_qv,
            "loss_bc_periodic": loss_bc_periodic,
            "loss_bc_blade": loss_bc_blade,
            "loss_phys": total,
            "q_hat_pred": torch.mean(q_hat_pred),
            "q_hat_target": torch.mean(q_hat_target),
        }


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
    g_star: float = 0.0,
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
        "g_star": float(g_star),
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
    g_star: float = 0.0,
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
                g_star=g_star,
            )
        )
    return cases


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
        data_weight: float = 1.0,
        physics_weight: float = 0.1,
        warmup_epochs: int = 20,
        ramp_epochs: int = 30,
        learn_ibm_params: bool = True,
        ibm_c_range: tuple[float, float] = (0.25, 4.0),
        ibm_epsilon_range: tuple[float, float] = (0.01, 0.08),
        ibm_hidden: int = 16,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_dataset = BladeFlowDataset(train_cases, input_mode=input_mode)
        self.val_dataset = BladeFlowDataset(val_cases if val_cases is not None else train_cases, input_mode=input_mode)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

        input_channels = self.train_dataset[0]["x"].shape[0]
        self.model = SliceWiseFNOFlowModel(
            input_channels=input_channels,
            modes=modes,
            width=width,
            depth=depth,
            z_padding=z_padding,
        ).to(self.device)
        self.model_config = {
            "input_channels": input_channels,
            "modes": modes,
            "width": width,
            "depth": depth,
            "z_padding": z_padding,
            "output_channels": 4,
        }
        default_ibm_c = float(np.mean([float(_pick(case, "ibm_C", default=1.0)) for case in train_cases]))
        default_ibm_epsilon = float(np.mean([float(_pick(case, "ibm_epsilon", default=0.025)) for case in train_cases]))
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
            "ibm_c_range": tuple(float(v) for v in ibm_c_range),
            "ibm_epsilon_range": tuple(float(v) for v in ibm_epsilon_range),
            "ibm_hidden": int(ibm_hidden),
            "default_ibm_c": default_ibm_c,
            "default_ibm_epsilon": default_ibm_epsilon,
        }
        self.trainer_config = {
            "input_mode": input_mode,
            "batch_size": batch_size,
            "lr": lr,
            "data_weight": data_weight,
            "physics_weight": physics_weight,
            "warmup_epochs": warmup_epochs,
            "ramp_epochs": ramp_epochs,
            "learn_ibm_params": self.learn_ibm_params,
            "ibm_c_range": tuple(float(v) for v in ibm_c_range),
            "ibm_epsilon_range": tuple(float(v) for v in ibm_epsilon_range),
            "ibm_hidden": int(ibm_hidden),
        }
        self.checkpoint_metadata: dict[str, Any] | None = None

        self.physics_loss = BladeFlowPhysicsLoss().to(self.device)
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

        print("\n========== SurrogateModeling: 数据准备摘要 ==========")
        print(f"device                : {self.device}")
        print(f"train_cases / val_cases: {len(self.train_dataset)} / {len(self.val_dataset)}")
        print(f"train_blade_geometries: {blade_geometry_count}")
        print(f"input_mode            : {self.train_dataset.input_mode}")
        print(f"learn_ibm_params      : {self.learn_ibm_params}")
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
            f"qv_hat={float(sample['qv_hat'].item()):.6g}"
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
            f"default_ibm           : C={float(sample['ibm_C'].item()):.6g}, "
            f"epsilon={float(sample['ibm_epsilon'].item()):.6g}"
        )
        if "blade_params" in case:
            print(f"sample_blade_params   : {case['blade_params']}")
        print("===============================================\n")

    def _current_ibm_params(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # 如果开启了自适应 IBM 参数，就按当前样本动态求一对 C / epsilon。
        # 否则退回到样本里保存的默认值。
        if self.ibm_mask_controller is None:
            ibm_c = batch["ibm_C"].view(-1, 1, 1, 1)
            ibm_epsilon = batch["ibm_epsilon"].view(-1, 1, 1, 1)
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

    def _prepare_runtime_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # 训练和部署阶段都使用“当前学习到的 IBM 参数”实时重建 phi 和 x。
        ibm_c, ibm_epsilon = self._current_ibm_params(batch)
        learned_phi = build_phi_from_signed_distance(batch["signed_distance"], ibm_c, ibm_epsilon)
        learned_phi = torch.where(batch["blade_mask"] > 0.5, torch.zeros_like(learned_phi), learned_phi)
        if "has_true_signed_distance" in batch:
            true_distance_mask = batch["has_true_signed_distance"].view(-1, 1, 1, 1) > 0.5
            phi = torch.where(true_distance_mask, learned_phi, batch["phi"])
            ibm_c = torch.where(true_distance_mask, ibm_c, batch["ibm_C"].view(-1, 1, 1, 1))
            ibm_epsilon = torch.where(
                true_distance_mask,
                ibm_epsilon,
                batch["ibm_epsilon"].view(-1, 1, 1, 1),
            )
        else:
            phi = learned_phi
        phi = torch.clamp(phi, min=0.0, max=1.0)
        phi = hard_project_theta_periodic(phi, theta_dim=2)

        runtime_batch = dict(batch)
        runtime_batch["phi"] = phi
        runtime_batch["x"] = self._compose_model_input(runtime_batch, phi)
        runtime_batch["ibm_C"] = ibm_c.view(-1)
        runtime_batch["ibm_epsilon"] = ibm_epsilon.view(-1)
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

        dataset = BladeFlowDataset([case], input_mode=self.train_dataset.input_mode)
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

            x_value = state[0] * np.cos(state[1])
            y_value = state[0] * np.sin(state[1])
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
        theme: str = "dark",
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
        ibm_c_value = float(sample["ibm_C"].item())
        ibm_epsilon_value = float(sample["ibm_epsilon"].item())

        train_n = int(self.train_dataset[0]["x"].shape[-1])
        deploy_n = int(sample["x"].shape[-1])
        bg = "#0E1117" if theme == "dark" else "white"
        blade_color = "#FFB000" if theme == "dark" else "#D55E00"
        passage_color = "lightgray"
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
            plotter.add_lines(root_line, color="steelblue", width=2)
            plotter.add_lines(tip_line, color="seagreen", width=2)

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

        colors = plt.cm.viridis(np.linspace(0.12, 0.95, max(len(base_seeds), 1)))[:, :3]
        phi_field = sample["phi"].detach().cpu().numpy()
        fields_phy_np = {name: tensor.detach().cpu().numpy() for name, tensor in pred_phy.items()}
        tube_radius = 0.0075 * config.delta_r

        streamline_count = 0
        for blade_id in range(passages_to_plot):
            theta_shift = blade_id * config.theta0
            for color, (seed_r, seed_theta, seed_z) in zip(colors, base_seeds):
                streamline = self._trace_streamline_cylindrical(
                    fields_phy=fields_phy_np,
                    phi_field=phi_field,
                    config=config,
                    seed_r=seed_r,
                    seed_theta=seed_theta + theta_shift,
                    seed_z=seed_z,
                    max_steps=max_streamline_steps,
                    step_scale=streamline_step_scale,
                )
                if streamline.shape[0] >= 2:
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
            f"blades={config.n_blade} | C={ibm_c_value:.4g} | eps={ibm_epsilon_value:.4g}",
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
        g_star: float = 0.0,
        input_mode: str = "both",
        batch_size: int = 1,
        lr: float = 1e-3,
        modes: int = 12,
        width: int = 32,
        depth: int = 4,
        z_padding: int = 8,
        learn_ibm_params: bool = True,
        ibm_c_range: tuple[float, float] = (0.25, 4.0),
        ibm_epsilon_range: tuple[float, float] = (0.01, 0.08),
        ibm_hidden: int = 16,
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
            g_star=g_star,
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
                g_star=g_star,
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
            data_weight=0.0,
            physics_weight=1.0,
            warmup_epochs=0,
            ramp_epochs=0,
            learn_ibm_params=learn_ibm_params,
            ibm_c_range=ibm_c_range,
            ibm_epsilon_range=ibm_epsilon_range,
            ibm_hidden=ibm_hidden,
            device=device,
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

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

        sample_weight = has_target.expand_as(pred["UR"])
        p_weight = torch.clamp(batch["phi"], min=0.0, max=1.0) * has_target

        loss_ur = self.physics_loss.weighted_mse(pred["UR"] - target_ur, sample_weight)
        loss_ut = self.physics_loss.weighted_mse(pred["UT"] - target_ut, sample_weight)
        loss_uz = self.physics_loss.weighted_mse(pred["UZ"] - target_uz, sample_weight)
        loss_p = self.physics_loss.weighted_mse(pred["P"] - target_p, p_weight)

        total = loss_ur + loss_ut + loss_uz + loss_p
        return total, {
            "loss_ur": loss_ur,
            "loss_ut": loss_ut,
            "loss_uz": loss_uz,
            "loss_p": loss_p,
            "loss_data": total,
        }

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

    def run_epoch(self, loader: DataLoader, epoch: int, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
            if self.ibm_mask_controller is not None:
                self.ibm_mask_controller.train()
        else:
            self.model.eval()
            if self.ibm_mask_controller is not None:
                self.ibm_mask_controller.eval()

        physics_factor = self.current_physics_factor(epoch)
        logs: dict[str, float] = {}
        count = 0

        for batch in loader:
            batch = self._to_device(batch)
            batch = self._prepare_runtime_batch(batch)

            with torch.set_grad_enabled(training):
                pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])
                loss_data, log_data = self.supervised_loss(pred, batch)
                loss_phys, log_phys = self.physics_loss(pred, batch)
                loss = self.data_weight * loss_data + physics_factor * loss_phys

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            merged = {
                "loss_total": loss,
                "physics_factor": torch.tensor(physics_factor, device=loss.device),
                "ibm_C": torch.mean(batch["ibm_C"]),
                "ibm_epsilon": torch.mean(batch["ibm_epsilon"]),
            }
            merged.update(log_data)
            merged.update(log_phys)

            for key, value in merged.items():
                logs[key] = logs.get(key, 0.0) + float(value.detach().cpu().item())
            count += 1

        return {key: value / max(count, 1) for key, value in logs.items()}

    def fit(self, epochs: int = 200, print_interval: int = 10) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            train_log = self.run_epoch(self.train_loader, epoch, True)
            val_log = self.run_epoch(self.val_loader, epoch, False)

            record: dict[str, float] = {}
            for key, value in train_log.items():
                record[f"train_{key}"] = value
            for key, value in val_log.items():
                record[f"val_{key}"] = value
            history.append(record)

            if epoch == 0 or (epoch + 1) % print_interval == 0:
                print(
                    f"Epoch {epoch + 1:04d} | "
                    f"train_total={train_log['loss_total']:.6e} | "
                    f"train_data={train_log['loss_data']:.6e} | "
                    f"train_phys={train_log['loss_phys']:.6e} | "
                    f"train_qv={train_log['loss_qv']:.6e} | "
                    f"train_ibm_C={train_log['ibm_C']:.6g} | "
                    f"train_ibm_eps={train_log['ibm_epsilon']:.6g} | "
                    f"train_bc_periodic={train_log['loss_bc_periodic']:.6e} | "
                    f"train_bc_blade={train_log['loss_bc_blade']:.6e} | "
                    f"val_total={val_log['loss_total']:.6e} | "
                    f"phys_factor={train_log['physics_factor']:.3e}"
                )

        return history

    def plot_training_history(
        self,
        history: Sequence[Mapping[str, float]],
        *,
        show: bool = True,
        save_path: str | Path | None = None,
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

        curve_specs: list[tuple[str, str]] = []
        if has_supervised_target:
            curve_specs.append(("loss_data", "Data"))
        curve_specs.extend(
            [
                ("loss_c", "R_c"),
                ("loss_r", "R_r"),
                ("loss_theta", "R_theta"),
                ("loss_z", "R_z"),
            ]
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
        panel_specs = [
            ("train_", "Training Loss History"),
            ("val_", "Validation Loss History"),
        ]

        for ax, (prefix, title) in zip(axes[0], panel_specs):
            plotted = False
            for key, label in curve_specs:
                full_key = f"{prefix}{key}"
                if full_key not in history[0]:
                    continue

                values = np.array([float(item.get(full_key, np.nan)) for item in history], dtype=float)
                if not np.any(np.isfinite(values)):
                    continue

                # 对数坐标不能直接画 0，这里只在绘图时做极小截断。
                values = np.where(np.isfinite(values), np.maximum(values, 1e-30), np.nan)
                ax.plot(epochs, values, linewidth=1.6, label=label)
                plotted = True

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
            if plotted:
                ax.legend()
            else:
                ax.text(0.5, 0.5, "No curves", transform=ax.transAxes, ha="center", va="center")

        fig.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
            print(f"训练损失对数曲线已保存到: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

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
        batch = self._to_device(batch)
        batch = self._prepare_runtime_batch(batch)
        pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])
        loss_data, log_data = self.supervised_loss(pred, batch)
        loss_phys, log_phys = self.physics_loss(pred, batch)
        physics_factor = self.current_physics_factor(0)
        loss = self.data_weight * loss_data + physics_factor * loss_phys

        if do_backward:
            self.optimizer.zero_grad()
            loss.backward()

        return {
            "loss_total": float(loss.detach().cpu().item()),
            "loss_data": float(log_data["loss_data"].detach().cpu().item()),
            "loss_phys": float(log_phys["loss_phys"].detach().cpu().item()),
            "loss_qv": float(log_phys["loss_qv"].detach().cpu().item()),
            "loss_bc_periodic": float(log_phys["loss_bc_periodic"].detach().cpu().item()),
            "loss_bc_blade": float(log_phys["loss_bc_blade"].detach().cpu().item()),
            "q_hat_pred": float(log_phys["q_hat_pred"].detach().cpu().item()),
            "q_hat_target": float(log_phys["q_hat_target"].detach().cpu().item()),
            "physics_factor": float(physics_factor),
            "ibm_C": float(batch["ibm_C"].detach().cpu().mean().item()),
            "ibm_epsilon": float(batch["ibm_epsilon"].detach().cpu().mean().item()),
            "has_target": float(batch["has_target"][0].detach().cpu().item()),
            "input_channels": float(batch["x"].shape[1]),
            "grid_size": float(batch["x"].shape[-1]),
        }

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
    ) -> list[dict[str, float]]:
        # 纯物理调试模式下，先看叶片导入，再训练，再看训练后的场。
        blade_plot_path = None
        post_plot_path = None
        post_3d_path = None
        history_plot_path = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            blade_plot_path = save_dir / "blade_spans.png"
            post_plot_path = save_dir / "post_physical_spans.png"
            post_3d_path = save_dir / "post_3d_streamlines.png"
            history_plot_path = save_dir / "training_loss_log.png"
            if save_checkpoint_path is None:
                save_checkpoint_path = save_dir / "surrogate_checkpoint.pt"
        if save_history_plot_path is not None:
            history_plot_path = Path(save_history_plot_path)

        print("纯物理调试模式：先展示叶片，再开始训练。")
        self.plot_blade_spans(case_index=0, spans=preview_spans, show=show_plots, save_path=blade_plot_path)
        history = self.fit(epochs=epochs, print_interval=print_interval)
        self.plot_training_history(history, show=show_plots, save_path=history_plot_path)
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
        ibm_c_value = float(sample["ibm_C"].item())
        ibm_epsilon_value = float(sample["ibm_epsilon"].item())

        print("\n========== 训练后 Post 检查 ==========")
        print(f"grid transfer: train_n={train_n}, deploy_n={deploy_n}")
        if deploy_n > train_n:
            print("当前展示的是“粗网格训练 -> 更细网格部署”的直接推理结果。")
        print(f"出口单流道体积流量: pred={q_pred:.6g}, target={q_target:.6g}")
        print(f"adaptive_ibm         : C={ibm_c_value:.6g}, epsilon={ibm_epsilon_value:.6g}")
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

            print(f"span={span:.2f} (i={r_index})")
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
                show=show,
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
            "ibm_C": ibm_c_value,
            "ibm_epsilon": ibm_epsilon_value,
            "train_n": train_n,
            "deploy_n": deploy_n,
            "three_d_info": three_d_info,
        }

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
        torch.save(
            {
                "checkpoint_version": 3,
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
            },
            str(path),
        )
        print(f"模型 checkpoint 已保存到: {path}")

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        ibm_state = payload.get("ibm_mask_controller_state_dict")
        if self.ibm_mask_controller is not None and ibm_state is not None:
            self.ibm_mask_controller.load_state_dict(ibm_state)
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
    ) -> "SurrogateModeling":
        # 用 checkpoint 重建一个可直接部署的新 trainer。
        path = Path(path)
        payload = torch.load(str(path), map_location="cpu")
        case_list = [cases] if isinstance(cases, Mapping) else list(cases)

        model_config = dict(payload.get("model_config", {}))
        trainer_config = dict(payload.get("trainer_config", {}))
        input_mode = payload.get("input_mode", trainer_config.get("input_mode", "both"))

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
            data_weight=float(trainer_config.get("data_weight", 0.0)),
            physics_weight=float(trainer_config.get("physics_weight", 1.0)),
            warmup_epochs=int(trainer_config.get("warmup_epochs", 0)),
            ramp_epochs=int(trainer_config.get("ramp_epochs", 0)),
            learn_ibm_params=bool(trainer_config.get("learn_ibm_params", True)),
            ibm_c_range=tuple(trainer_config.get("ibm_c_range", (0.25, 4.0))),
            ibm_epsilon_range=tuple(trainer_config.get("ibm_epsilon_range", (0.01, 0.08))),
            ibm_hidden=int(trainer_config.get("ibm_hidden", 16)),
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
        )


def load_cases_from_pt(path: str | Path) -> list[Mapping[str, Any]]:
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, dict) and "cases" in payload:
        return list(payload["cases"])
    return list(payload)


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


if __name__ == "__main__":
    # 这里保留一个最直接的纯物理调试入口：
    # 自动寻找 blade_params.json，找到就跑 smoke test 和纯物理训练。
    seed_everything(42)
    blade_params = find_first_blade_params("../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc")
    if blade_params is not None:
        trainer = SurrogateModeling.build_pure_physics_debug_trainer(
            blade_params=blade_params,
            n=48,
            batch_size=1,
            mu=0.006,   # LBE动力粘度
            rho=10650.0,   # LBE密度
            omega=-210.0 * 2 * np.pi / 60.0,  # 转速210rpm
            qv=0.16,   # 出口体积流量0.16m3/s
            lr=1e-3,   # 学习率
            learn_ibm_params=True,    # 是否将IBM参数也纳入学习范围
            ibm_c_range=(0.3, 3.0),
            ibm_epsilon_range=(0.001, 0.05),
        )
        '''
        可以传入列表blade_params（训练集），val_blades为测试集
        trainer = SurrogateModeling.build_pure_physics_debug_trainer(
            blade_params=train_blades,
            val_blade_params=val_blades,
            n=48,
            mu=0.006,
            rho=10650.0,
            omega=-210.0 * 2 * np.pi / 60.0,
            qv=0.16,
        )
        '''
        smoke = trainer.smoke_test(do_backward=True)
        print("Smoke test:", smoke)
        trainer.fit_pure_physics_debug(
            epochs=10000,
            print_interval=1,
            preview_spans=(0.4, 0.6, ),
            post_spans=(0.4, 0.6, ),
            show_plots=True,
            save_dir="surrogate_debug_outputs",
            save_checkpoint_path="surrogate_debug_outputs/surrogate_checkpoint.pt",
            plot_3d=True,
        )

        # 体现“粗网格训练、细网格部署”，直接再构一个更细网格的部署工况：
        fine_case = make_pure_physics_debug_case(
            blade_params=blade_params,
            n=96,
            mu=0.006,
            rho=10650.0,
            omega=-210.0 * 2 * np.pi / 60.0,   # 转速210rpm
            qv=0.16,
        )
        SurrogateModeling.deploy_from_checkpoint(
            "surrogate_debug_outputs/surrogate_checkpoint.pt",
            fine_case,
            show=True,
            save_path_2d="surrogate_debug_outputs/fine_grid_spans.png",
            save_path_3d="surrogate_debug_outputs/fine_grid_3d_streamlines.png",
            spans=(0.4, 0.6),
            plot_3d=True,
        )
    else:
        print("No blade_params.json found. Use build_pure_physics_debug_trainer(...) or load_cases_from_pt(...).")
