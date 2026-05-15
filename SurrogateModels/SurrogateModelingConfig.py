from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from BladeImport import PassageGeometry
from SurrogateModelingUtils import _pick, _scalar_from_any


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
            ibm_C=_scalar_from_any(
                _pick(case, "ibm_C_profile", "ibm_C_span", "ibm_C_r", "ibm_C", default=1.0),
                default=1.0,
            ),
            ibm_epsilon=_scalar_from_any(
                _pick(
                    case,
                    "ibm_epsilon_profile",
                    "ibm_epsilon_span",
                    "ibm_epsilon_r",
                    "ibm_epsilon",
                    default=0.025,
                ),
                default=0.025,
            ),
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
        return self.rho * (self.u_zo ** 2 + self.u_omega ** 2 / 2.0 + self.g * self.h)

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
    def g_star(self) -> float:
        # Dimensionless Document.pdf, Eq. (2.46):
        # G* = g * DeltaR / (u_omega * u_z,o)
        # g is the physical gravity input; G* is always derived from the current scales.
        denom = self.u_omega * self.u_zo
        if abs(denom) < 1e-12:
            return 0.0
        return self.g * self.delta_r / denom

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
