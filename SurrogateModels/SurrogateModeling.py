from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import NeuralOperators
from BladeImport import PassageGeometry, build_blade_boundary


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
    ibm_C: float,
    ibm_epsilon: float,
) -> torch.Tensor:
    # 这里和 DataGenerator 中的浸没边界写法保持一致：
    # phi=1 近似纯流体，phi=0 近似纯固体，中间是平滑过渡层。
    return 1.0 - torch.exp(-ibm_C * signed_distance ** 2 / (ibm_epsilon ** 2))


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
        self.core = NeuralOperators.CNO2d_small(
            modes=modes,
            # cheb_modes=(modes, modes),
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

    @staticmethod
    def _expand_scalar(x: torch.Tensor) -> torch.Tensor:
        return x.view(-1, 1, 1, 1)

    @staticmethod
    def _line_quadrature_weight(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # 当前 R / Theta 网格都包含首尾端点，因此做积分时采用梯形权重。
        weight = torch.ones(length, device=device, dtype=dtype)
        if length > 1:
            weight[0] = 0.5
            weight[-1] = 0.5
        return weight

    @staticmethod
    def _neighbor_plus(x: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
        if periodic:
            return torch.roll(x, shifts=-1, dims=dim)
        out = torch.roll(x, shifts=-1, dims=dim)
        index = [slice(None)] * x.ndim
        index[dim] = -1
        out[tuple(index)] = x[tuple(index)]
        return out

    @staticmethod
    def _neighbor_minus(x: torch.Tensor, dim: int, periodic: bool) -> torch.Tensor:
        if periodic:
            return torch.roll(x, shifts=1, dims=dim)
        out = torch.roll(x, shifts=1, dims=dim)
        index = [slice(None)] * x.ndim
        index[dim] = 0
        out[tuple(index)] = x[tuple(index)]
        return out

    @staticmethod
    def _d1_periodic_with_overlap(
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        # Theta 方向使用“首尾重合网格”。
        # 因此在 seam 点求导时，左邻点应该取倒数第二个点，而不是最后一个重合点。
        x_perm = torch.movedim(x, dim, -1)
        out = torch.zeros_like(x_perm)
        n = x_perm.shape[-1]

        if n <= 1:
            return torch.movedim(out, -1, dim)

        if n == 2:
            seam = (x_perm[..., 1] - x_perm[..., 0]) / torch.clamp(spacing, min=1e-12)
            out[..., 0] = seam
            out[..., 1] = seam
            return torch.movedim(out, -1, dim)

        out[..., 1:-1] = (x_perm[..., 2:] - x_perm[..., :-2]) / (2.0 * spacing)
        seam = (x_perm[..., 1] - x_perm[..., -2]) / (2.0 * spacing)
        out[..., 0] = seam
        out[..., -1] = seam
        return torch.movedim(out, -1, dim)

    @staticmethod
    def _d2_periodic_with_overlap(
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        x_perm = torch.movedim(x, dim, -1)
        out = torch.zeros_like(x_perm)
        n = x_perm.shape[-1]

        if n <= 1:
            return torch.movedim(out, -1, dim)

        if n == 2:
            seam = (x_perm[..., 1] - 2.0 * x_perm[..., 0] + x_perm[..., 1]) / (spacing ** 2)
            out[..., 0] = seam
            out[..., 1] = seam
            return torch.movedim(out, -1, dim)

        out[..., 1:-1] = (x_perm[..., 2:] - 2.0 * x_perm[..., 1:-1] + x_perm[..., :-2]) / (spacing ** 2)
        seam = (x_perm[..., 1] - 2.0 * x_perm[..., 0] + x_perm[..., -2]) / (spacing ** 2)
        out[..., 0] = seam
        out[..., -1] = seam
        return torch.movedim(out, -1, dim)

    def d1(
        self,
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
        periodic: bool,
        duplicate_endpoint: bool = False,
    ) -> torch.Tensor:
        if periodic and duplicate_endpoint:
            return self._d1_periodic_with_overlap(x, dim, spacing)
        xp = self._neighbor_plus(x, dim, periodic)
        xm = self._neighbor_minus(x, dim, periodic)
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
            return self._d2_periodic_with_overlap(x, dim, spacing)
        xp = self._neighbor_plus(x, dim, periodic)
        xm = self._neighbor_minus(x, dim, periodic)
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

        theta_weight = self._line_quadrature_weight(mask.shape[2], mask.device, mask.dtype)
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
        dR = self._expand_scalar(batch["dR"])
        dTheta = self._expand_scalar(batch["dTheta"])
        dZ = self._expand_scalar(batch["dZ"])
        Lambda = self._expand_scalar(batch["Lambda"])

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

        r_weight = self._line_quadrature_weight(uz.shape[1], uz.device, uz.dtype)
        theta_weight = self._line_quadrature_weight(uz.shape[2], uz.device, uz.dtype)
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

        dR = self._expand_scalar(batch["dR"])
        dTheta = self._expand_scalar(batch["dTheta"])
        dZ = self._expand_scalar(batch["dZ"])
        Eu = self._expand_scalar(batch["Eu_omega"])
        Re = self._expand_scalar(batch["Re_omega"])
        Lambda = self._expand_scalar(batch["Lambda"])
        Ku = self._expand_scalar(batch["Ku"])
        delta = self._expand_scalar(batch["delta"])
        sgn_omega = self._expand_scalar(batch["sgn_omega"])
        g_star = self._expand_scalar(batch["g_star"])
        absolute_frame = self._expand_scalar(batch["absolute_frame"])

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

        self.physics_loss = BladeFlowPhysicsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs

        self.print_preparation_summary(case_index=0)

    def print_preparation_summary(self, case_index: int = 0) -> None:
        # 打印一次样本准备摘要，方便确认几何、尺度和边界设置是否符合预期。
        sample = self.train_dataset[case_index]
        case = self.train_dataset.cases[case_index]

        blade_cells = int(sample["blade_mask"].sum().item())
        total_cells = int(sample["blade_mask"].numel())
        fluid_ratio = float(sample["phi"].mean().item())
        has_target = bool(sample["has_target"].item() > 0.5)

        print("\n========== SurrogateModeling: 数据准备摘要 ==========")
        print(f"device                : {self.device}")
        print(f"train_cases / val_cases: {len(self.train_dataset)} / {len(self.val_dataset)}")
        print(f"input_mode            : {self.train_dataset.input_mode}")
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
        if "blade_params" in case:
            print(f"blade_params          : {case['blade_params']}")
        print("===============================================\n")

    @staticmethod
    def _field_stats(field: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
        values = field[mask]
        if values.numel() == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(values.mean().item()),
            float(values.min().item()),
            float(values.max().item()),
        )

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
        mask = sample["blade_mask"].detach().cpu().numpy()
        phi = sample["phi"].detach().cpu().numpy()
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
        input_mode: str = "both",
        batch_size: int = 1,
        lr: float = 1e-3,
        modes: int = 12,
        width: int = 32,
        depth: int = 4,
        z_padding: int = 8,
        device: str = "cuda",
    ) -> "SurrogateModeling":
        case = make_pure_physics_debug_case(
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
        return cls(
            train_cases=[case],
            val_cases=[case],
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
        else:
            self.model.eval()

        physics_factor = self.current_physics_factor(epoch)
        logs: dict[str, float] = {}
        count = 0

        for batch in loader:
            batch = self._to_device(batch)

            with torch.set_grad_enabled(training):
                pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])
                loss_data, log_data = self.supervised_loss(pred, batch)
                loss_phys, log_phys = self.physics_loss(pred, batch)
                loss = self.data_weight * loss_data + physics_factor * loss_phys

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            merged = {"loss_total": loss, "physics_factor": torch.tensor(physics_factor, device=loss.device)}
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
                    f"train_bc_periodic={train_log['loss_bc_periodic']:.6e} | "
                    f"train_bc_blade={train_log['loss_bc_blade']:.6e} | "
                    f"val_total={val_log['loss_total']:.6e} | "
                    f"phys_factor={train_log['physics_factor']:.3e}"
                )

        return history

    def smoke_test(self, do_backward: bool = True) -> dict[str, float]:
        # 最小化检查一遍：
        # 1. DataLoader 是否正常
        # 2. 前向传播是否正常
        # 3. 数据损失 / 物理损失是否都能算
        # 4. backward 是否能打通
        batch = next(iter(self.train_loader))
        batch = self._to_device(batch)

        self.model.train()
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
    ) -> list[dict[str, float]]:
        # 纯物理调试模式下，先看叶片导入，再训练，再看训练后的场。
        blade_plot_path = None
        post_plot_path = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            blade_plot_path = save_dir / "blade_spans.png"
            post_plot_path = save_dir / "post_physical_spans.png"

        print("纯物理调试模式：先展示叶片，再开始训练。")
        self.plot_blade_spans(case_index=0, spans=preview_spans, show=show_plots, save_path=blade_plot_path)
        history = self.fit(epochs=epochs, print_interval=print_interval)
        self.post_process_case(case_index=0, spans=post_spans, show=show_plots, save_path=post_plot_path)
        return history

    @torch.no_grad()
    def predict_case(
        self,
        case: Mapping[str, Any],
        return_physical: bool = False,
    ) -> dict[str, torch.Tensor]:
        dataset = BladeFlowDataset([case], input_mode=self.train_dataset.input_mode)
        batch = dataset[0]
        batch = {key: value.unsqueeze(0).to(self.device) for key, value in batch.items()}

        pred = self.model(batch["x"], batch["phi"], batch["solid_ut"])

        if not return_physical:
            return {key: value[0].detach().cpu() for key, value in pred.items()}

        u_omega = float(batch["u_omega"][0].cpu().item())
        u_zo = float(batch["u_zo"][0].cpu().item())
        p0 = float(batch["P0"][0].cpu().item())
        return {
            "UR": pred["UR"][0].detach().cpu() * u_omega,
            "UT": pred["UT"][0].detach().cpu() * u_omega,
            "UZ": pred["UZ"][0].detach().cpu() * u_zo,
            "P": pred["P"][0].detach().cpu() * p0,
        }

    @torch.no_grad()
    def post_process_case(
        self,
        case_index: int = 0,
        spans: Sequence[float] = (0.2, 0.5, 0.8),
        *,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> dict[str, torch.Tensor]:
        # 把预测场映回物理量，并做几个最直观的检查。
        case = self.train_dataset.cases[case_index]
        sample = self.train_dataset[case_index]
        pred_dim = self.predict_case(case, return_physical=False)
        pred_phy = self.predict_case(case, return_physical=True)

        batch = {key: value.unsqueeze(0).to(self.device) for key, value in sample.items()}
        pred_batch = {key: value.unsqueeze(0).to(self.device) for key, value in pred_dim.items()}
        q_pred = float(self.physics_loss.outlet_flow_rate(pred_batch["UZ"], batch).detach().cpu().item())
        q_target = float(sample["qv_passage"].item())

        print("\n========== 训练后 Post 检查 ==========")
        print(f"出口单流道体积流量: pred={q_pred:.6g}, target={q_target:.6g}")
        print("下面给出各个 span 处流体区域的 mean / min / max，单位已经回到物理空间。")
        mask = sample["blade_mask"] < 0.5
        n = mask.shape[0]
        for span in spans:
            r_index = span_to_index(span, n)
            fluid = mask[r_index]
            ur_stats = self._field_stats(pred_phy["UR"][r_index], fluid)
            ut_stats = self._field_stats(pred_phy["UT"][r_index], fluid)
            uz_stats = self._field_stats(pred_phy["UZ"][r_index], fluid)
            p_stats = self._field_stats(pred_phy["P"][r_index], fluid)

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

        return pred_phy

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "input_mode": self.train_dataset.input_mode,
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])


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
        )
        smoke = trainer.smoke_test(do_backward=True)
        print("Smoke test:", smoke)
        trainer.fit_pure_physics_debug(
            epochs=10000,
            print_interval=1,
            preview_spans=(0.4, 0.6, ),
            post_spans=(0.4, 0.6, ),
            show_plots=True,
            save_dir="surrogate_debug_outputs",
        )
    else:
        print("No blade_params.json found. Use build_pure_physics_debug_trainer(...) or load_cases_from_pt(...).")
