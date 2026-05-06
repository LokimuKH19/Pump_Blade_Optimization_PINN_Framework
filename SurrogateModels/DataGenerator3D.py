from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from BladeImport import attach_blade_to_solver
import matplotlib.pyplot as plt
import numpy as np
from PressureUpdaters3D import PressureUpdater3D
import torch


@dataclass
class SolveLog3D:
    method: str
    converged: bool
    iterations: int
    final_momentum: float
    final_mass: float
    final_update: float
    final_q_hat: float
    delta_p: float


class BladeCalc3D:
    """三维无量纲柱坐标 DataGenerator。

    本文件参考 DataGenerator.py 的结构，但不再把 R 方向当成 batch。求解变量为
    [UR, UT, UZ, P]，坐标为 R/Theta/Z，三维连续性和三维 FVM 六面通量均参与求解。

    对应 Dimensionless Document.pdf 的主要实现点：
    - 式 (2.22)-(2.31)：几何、速度、压力无量纲化；
    - 式 (2.49)-(2.52)：三维连续性和三方向动量残差；
    - 式 (2.69)-(2.85)：动量方程 FVM 线性化；
    - 式 (2.86)-(2.96)：SIMPLE 压力修正使用局部矩阵逆/Schur complement；
    - 外层使用出口 q_hat 调整入口总压差 delta_p_global。

    说明：默认使用旋转参考系和完整 R-Theta 局部耦合；A12/A21、Rhie-Chow 与
    压力修正 Schur complement 保持同一套局部矩阵闭合。
    """

    def __init__(
        self,
        n: int = 48,
        rh: float = 0.0605,
        rs: float = 0.08,
        h: float = 0.125,
        mu: float = 0.006,
        rho: float = 10650.0,
        omega: float = -420.0 * np.pi / 60.0,
        qv: float = 0.16,
        n_blade: int = 6,
        max_iter: int = 400,
        tol: float = 1e-5,
        u_relax: float = 0.35,
        p_relax: float = 0.20,
        device: str = "cuda",
        z0: float = 0.0,
        blade_params: str | Path | None = "../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc/blade_params.json",
        absolute_frame: bool = False,
        delta_p_initial: float = 0.1,
        pressure_solver: str = "gmg",
        pressure_max_inner: int = 500,
        pressure_tol: float = 1e-7,
        ibm_C: float = 1.0,
        ibm_epsilon: float = 0.025,
        adaptive_ibm: bool = True,
        ibm_band_cells: float = 1.5,
        ibm_adapt_interval: int = 8,
        ibm_adapt_relax: float = 0.20,
        startup_nontrivial: bool = True,
        initialization: str = "fluent_hybrid",
        initialization_sweeps: int = 8,
        initialization_swirl_fraction: float = 1.0,
        initialization_wake_strength: float = 0.45,
        flow_boundary_mode: str = "outlet_flow",
        balance_inlet_flow: bool = True,
        flow_boundary_profile_floor: float = 0.04,
        pseudo_dt: float = 5e-4,
        local_pseudo_dt: bool = True,
        pseudo_cfl: float = 1.2,
        rans_model: str = "mixing_length",
        rans_mixing_length: float = 0.03,
        rans_nut_max_ratio: float = 20.0,
        cross_coupling_strength: float = 1.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ----- Physical and dimensionless scales, matching PDF section 1.1.2.
        self.g = 9.8
        self.n = int(n)
        if self.n < 24:
            print(
                f"Warning: n={self.n} is a debugging grid. "
                "3D IBM and pressure projection behavior on this grid is not production-representative."
            )
        self.rh = float(rh)
        self.rs = float(rs)
        self.h = float(h)
        self.z0 = float(z0)
        self.mu = float(mu)
        self.rho = float(rho)
        self.nu = self.mu / self.rho
        self.omega = float(omega)
        self.omega_abs = max(abs(self.omega), 1e-12)
        self.sgn_omega = 1.0 if self.omega >= 0.0 else -1.0
        self.qv = float(qv)
        self.n_blade = int(n_blade)
        self.n_blades = self.n_blade
        self.theta0 = 2.0 * np.pi / self.n_blade
        self.blade_params = None if blade_params is None else str(blade_params)
        self.absolute_frame = bool(absolute_frame)

        self.delta_r = self.rs - self.rh
        if self.delta_r <= 0.0:
            raise ValueError("rs must be larger than rh.")
        self.u_omega = self.rs * self.omega_abs
        self.u_zo = self.qv / (np.pi * (self.rs**2 - self.rh**2))
        self.P0 = self.rho * (0.5 * self.u_omega**2 + 0.5 * self.u_zo**2 + self.g * self.h)
        self.Re_omega = self.u_omega * self.delta_r / max(self.nu, 1e-30)
        self.Eu_omega = self.P0 / (self.rho * self.u_omega**2)
        self.Lambda = self.delta_r / self.h
        self.Ku = self.u_zo / self.u_omega
        self.delta = self.delta_r / self.rs
        self.G_star = self.g * self.delta_r / (self.u_omega * max(self.u_zo, 1e-30))
        self.qv_passage = self.qv / self.n_blade
        self.qv_hat = self.qv_passage / (self.u_zo * self.delta_r**2 * self.theta0)

        # ----- Iteration controls.
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.u_relax = float(u_relax)
        self.p_relax = float(p_relax)
        self.delta_p_global = float(delta_p_initial)
        self.delta_p_min = 1e-9
        self.delta_p_max = 1e8
        self.outer_flow_tol = 2e-3
        self.delta_p_relax = 0.50
        self.pressure_solver = str(pressure_solver).lower()
        self.pressure_max_inner = int(pressure_max_inner)
        self.pressure_tol = float(pressure_tol)
        self.pseudo_dt = max(float(pseudo_dt), 1e-8)
        self.local_pseudo_dt = bool(local_pseudo_dt)
        self.pseudo_cfl = max(float(pseudo_cfl), 1e-6)

        # ----- Stabilization knobs for the pseudo-transient COUPLE algorithm.
        self.couple_relax = 0.18
        self.momentum_sweeps = 8
        self.momentum_solver_relax = 0.65
        self.couple_pressure_sweeps = 1
        self.couple_pressure_interval = 1
        self.pressure_projection_relax = 0.50
        self.couple_pressure_backtracking = True
        self.couple_pressure_min_relax = 1e-3
        self.couple_pressure_max_momentum_growth = 1.50
        self.couple_pressure_momentum_allowance = 2e-2
        self.couple_pressure_min_mass_drop_on_momentum_growth = 0.05
        self.couple_momentum_update_limit = 0.15
        self.couple_pressure_velocity_limit = 0.18
        self.couple_pressure_update_limit = 0.75
        self.couple_field_abs_limit = 40.0
        self.couple_flow_control = True
        self.couple_flow_control_interval = 20
        self.couple_flow_control_gain = 0.20
        self.couple_flow_control_clip = 0.08

        # ----- GMG controls; PressureUpdater3D reads these directly.
        self.gmg_levels = 5
        self.gmg_pre_smooth = 3
        self.gmg_post_smooth = 3
        self.gmg_cycles = 6
        self.gmg_omega = 0.70

        # ----- IBM and simple zero-equation RANS settings.
        self.ibm_C = float(ibm_C)
        self.ibm_epsilon = float(ibm_epsilon)
        self.adaptive_ibm = bool(adaptive_ibm)
        self.ibm_band_cells = max(float(ibm_band_cells), 0.5)
        self.ibm_adapt_interval = max(int(ibm_adapt_interval), 1)
        self.ibm_adapt_relax = float(np.clip(ibm_adapt_relax, 0.0, 1.0))
        self.startup_nontrivial = bool(startup_nontrivial)
        self.initialization = "none" if not self.startup_nontrivial else str(initialization).lower()
        self.initialization_sweeps = max(int(initialization_sweeps), 0)
        self.initialization_swirl_fraction = max(float(initialization_swirl_fraction), 0.0)
        self.initialization_wake_strength = max(float(initialization_wake_strength), 0.0)
        self.flow_boundary_mode = self._normalize_flow_boundary_mode(flow_boundary_mode)
        self.balance_inlet_flow = bool(balance_inlet_flow)
        self.flow_boundary_profile_floor = max(float(flow_boundary_profile_floor), 1e-6)
        self._ibm_adapt_count = 0
        self.ibm_hard_phi = 0.05
        self.rans_model = str(rans_model).lower()
        self.rans_mixing_length = max(float(rans_mixing_length), 0.0)
        self.rans_nut_max_ratio = max(float(rans_nut_max_ratio), 0.0)
        self.rans_smoothing_steps = 1
        self.cross_coupling_strength = max(float(cross_coupling_strength), 0.0)
        self.coupling_det_fraction = 0.35

        self._build_grid()
        self._build_geometry()
        self._init_fields()
        self._init_blade_boundary()
        self._initialize_physical_startup()
        self._apply_boundary()

        # 动量局部矩阵字段。A12/A21 默认可以是 0，但压力更新器仍按完整矩阵逆处理。
        self.A11 = torch.ones_like(self.P)
        self.A12 = torch.zeros_like(self.P)
        self.A21 = torch.zeros_like(self.P)
        self.A22 = torch.ones_like(self.P)
        self.A33 = torch.ones_like(self.P)
        self.P_prime = torch.zeros_like(self.P)
        self.pressure_updater = PressureUpdater3D(self)

        # Predictor fields used by SIMPLE/COUPLE.
        self.UR_tilde = self.UR.clone()
        self.UT_tilde = self.UT.clone()
        self.UZ_tilde = self.UZ.clone()
        self.nut_ratio = torch.zeros_like(self.P)
        self.last_history: list[dict[str, float]] = []
        self.iteration_history: list[dict[str, float]] = []
        self.last_solve_method = ""
        self.last_pressure_projection: dict[str, float] = {}
        self._couple_step_count = 0

    def _build_grid(self) -> None:
        n = self.n
        if n < 4:
            raise ValueError("n must be at least 4 in each direction.")
        self.dR = 1.0 / (n - 1)
        self.dTheta = 1.0 / (n - 1)
        self.dZ = 1.0 / (n - 1)
        r = torch.linspace(0.0, 1.0, n, device=self.device)
        theta = torch.linspace(0.0, 1.0, n, device=self.device)
        z = torch.linspace(0.0, 1.0, n, device=self.device)
        rr, tt, zz = torch.meshgrid(r, theta, z, indexing="ij")
        self.R = rr
        self.Theta = tt
        self.Z = zz
        self.r_hatC = rr + self.rh / self.delta_r
        self.K_theta_C = 1.0 / torch.clamp(self.r_hatC * self.theta0, min=1e-12)

    def _build_geometry(self) -> None:
        # FVM 中所有量都已经无量纲化，下面是控制体体积和各面面积的无量纲权重。
        self.dV = torch.full_like(self.r_hatC, self.dR * self.dTheta * self.dZ)
        self.radial_area = torch.full_like(self.r_hatC, self.dTheta * self.dZ)
        self.theta_area = torch.full_like(self.r_hatC, self.dR * self.dZ)
        self.z_area = torch.full_like(self.r_hatC, self.dR * self.dTheta)
        self.r_hat_E = 0.5 * (self.r_hatC + self.neighbor_raw(self.r_hatC, "E"))
        self.r_hat_W = 0.5 * (self.r_hatC + self.neighbor_raw(self.r_hatC, "W"))
        self.r_hat_E[-1, :, :] = self.r_hatC[-1, :, :]
        self.r_hat_W[0, :, :] = self.r_hatC[0, :, :]

        r_weight = torch.ones(self.n, device=self.device)
        theta_weight = torch.ones(self.n, device=self.device)
        r_weight[0] = r_weight[-1] = 0.5
        theta_weight[0] = theta_weight[-1] = 0.5
        self.r_quad_weight = r_weight.view(-1, 1)
        self.theta_quad_weight = theta_weight.view(1, -1)
        self.outlet_area_weight = (
            self.r_hatC[:, :, -1] * self.r_quad_weight * self.theta_quad_weight * self.dR * self.dTheta
        )
        self.outlet_area_target = torch.sum(self.outlet_area_weight)

    def _init_fields(self) -> None:
        shape = self.r_hatC.shape
        self.P = self.delta_p_global * (1.0 - self.Z)
        self.UR = torch.zeros(shape, device=self.device)
        self.UT = torch.zeros(shape, device=self.device)
        self.UZ = torch.ones(shape, device=self.device)
        self.blade_mask = None
        self.blade_distance = None
        self.blade_distance_z = None
        self.blade_footprint = None
        self.blade_boundary_meta = None
        self.blade_boundary_band_cells = self.ibm_band_cells
        self.boundary = None
        self.phi = torch.ones(shape, device=self.device)
        self.phi_mask = self.phi

    def _init_blade_boundary(self) -> None:
        if self.blade_params is None:
            return
        path = Path(self.blade_params)
        if not path.exists():
            return
        self.boundary = self.attach_blade_boundary(path, self.blade_boundary_band_cells)
        if self.adaptive_ibm:
            self.ibm_C, self.ibm_epsilon = self._ibm_target_parameters()
        self._refresh_ibm_phi()

    def _ibm_target_parameters(
        self,
        mass_error: float | None = None,
        momentum_error: float | None = None,
    ) -> tuple[float, float]:
        h_grid = max(min(self.dR, self.dTheta, self.dZ), 1e-12)
        eps = float(np.clip(self.ibm_band_cells * h_grid, 0.75 * h_grid, 3.5 * h_grid))
        target_phi = 0.70
        if mass_error is not None and np.isfinite(mass_error):
            target_phi += 0.08 * float(np.clip(np.log10(max(mass_error, 1e-12) / max(self.tol, 1e-12)), 0.0, 2.0)) / 2.0
        if momentum_error is not None and np.isfinite(momentum_error):
            target_phi -= 0.04 * float(np.clip(np.log10(max(momentum_error, 1e-12) / 1e-3), 0.0, 2.0)) / 2.0
        target_phi = float(np.clip(target_phi, 0.55, 0.86))
        c_value = -np.log(max(1.0 - target_phi, 1e-6)) * (eps / h_grid) ** 2
        return float(np.clip(c_value, 0.35, 12.0)), eps

    def _refresh_ibm_phi(self) -> None:
        if self.blade_distance is None:
            self.phi = torch.ones_like(self.P)
            self.phi_mask = self.phi
            return
        signed_distance = torch.as_tensor(self.blade_distance, dtype=torch.float32, device=self.device)
        eps2 = max(self.ibm_epsilon**2, 1e-20)
        self.phi = 1.0 - torch.exp(-self.ibm_C * signed_distance**2 / eps2)
        if self.blade_mask is not None:
            self.phi = torch.where(self.blade_mask, torch.zeros_like(self.phi), self.phi)
        self.phi = torch.clamp(self.phi, 0.0, 1.0)
        self.phi_mask = self.phi

    def _adapt_ibm(self, mass_error: float | None = None, momentum_error: float | None = None) -> None:
        if not self.adaptive_ibm or self.blade_distance is None:
            return
        self._ibm_adapt_count += 1
        if self._ibm_adapt_count % self.ibm_adapt_interval != 0:
            return
        target_c, target_eps = self._ibm_target_parameters(mass_error, momentum_error)
        relax = self.ibm_adapt_relax
        self.ibm_C = (1.0 - relax) * self.ibm_C + relax * target_c
        self.ibm_epsilon = (1.0 - relax) * self.ibm_epsilon + relax * target_eps
        self._refresh_ibm_phi()
        self._apply_boundary()

    def _scale_initial_axial_flow(self) -> None:
        fluid = torch.clamp(self.phi[:, :, -1], min=0.0, max=1.0)
        q_raw = torch.sum(self.UZ[:, :, -1] * fluid * self.outlet_area_weight)
        q_target = torch.sum(fluid * self.outlet_area_weight)
        if torch.isfinite(q_raw) and torch.abs(q_raw) > 1e-12 and torch.isfinite(q_target):
            scale = torch.clamp(q_target / q_raw, min=0.15, max=8.0)
            self.UZ = self.UZ * scale

    def _initial_profiles(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_core = torch.clamp(4.0 * self.R * (1.0 - self.R), min=0.0, max=1.0)
        wall_damp = torch.sqrt(torch.clamp(r_core, min=0.0, max=1.0))
        opening = torch.clamp(self.phi, min=0.0, max=1.0)
        wake = torch.clamp(1.0 - opening, min=0.0, max=1.0)
        axial_shape = 0.12 + 1.35 * r_core
        axial_shape = axial_shape * (1.0 - self.initialization_wake_strength * wake)
        axial_shape = torch.clamp(axial_shape, min=0.03)

        if self.absolute_frame:
            swirl = self.initialization_swirl_fraction * self._solid_ut() * (1.0 - wall_damp)
        else:
            # In a rotating frame, quiescent absolute inlet flow appears with -Omega*r tangential speed.
            swirl = -self.sgn_omega * self.delta * self.r_hatC
            swirl = self.initialization_swirl_fraction * swirl * wall_damp * (0.35 + 0.65 * opening)

        theta_deflect = torch.tanh(0.20 * self.K_theta_C * self._pressure_gradient_theta(opening))
        radial_deflect = torch.tanh(0.20 * self._pressure_gradient_r(opening))
        z_ramp = torch.sin(np.pi * torch.clamp(self.Z, min=0.0, max=1.0)) ** 2
        ur = -0.10 * self.initialization_wake_strength * radial_deflect * z_ramp
        ut = swirl - 0.16 * self.initialization_wake_strength * theta_deflect * z_ramp
        uz = axial_shape
        return ur, ut, uz

    def _apply_initialization_boundary_to(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        ut_inlet: torch.Tensor,
        uz_inlet: torch.Tensor,
    ) -> None:
        solid_ut = self._solid_ut()
        hard = self._hard_solid_mask()
        ur[:, :, 0] = 0.0
        ur[:, :, -1] = ur[:, :, -2]
        ut[:, :, 0] = ut_inlet[:, :, 0]
        ut[:, :, -1] = ut[:, :, -2]
        uz[:, :, 0] = uz_inlet[:, :, 0]
        uz[:, :, -1] = uz[:, :, -2]
        self._apply_flow_boundary_to(uz)
        ur[0, :, :] = 0.0
        ur[-1, :, :] = 0.0
        uz[0, :, :] = 0.0
        uz[-1, :, :] = 0.0
        ut[0, :, :] = solid_ut[0, :, :]
        ut[-1, :, :] = solid_ut[-1, :, :]
        ur[:] = torch.where(hard, torch.zeros_like(ur), ur)
        ut[:] = torch.where(hard, solid_ut, ut)
        uz[:] = torch.where(hard, torch.zeros_like(uz), uz)
        self._project_theta_periodic(ur, ut, uz)

    # 参考fluent的风格先跑一个initialization
    def initialize_flow(self, method: str | None = None, sweeps: int | None = None) -> dict[str, float]:
        """Initialize the 3D field with Fluent-like standard/hybrid initialization.

        `standard` patches the whole domain from inlet/rotating-frame estimates.
        `fluent_hybrid` additionally solves a lightweight Laplace interpolation so the
        interior is initialized from boundaries and blade blockage instead of a zero field.
        """
        method_name = (method or self.initialization).lower().replace("-", "_")
        if method_name in {"none", "off", "zero"}:
            return self.initialization_summary()

        ur0, ut0, uz0 = self._initial_profiles()
        self.UR = ur0.clone()
        self.UT = ut0.clone()
        self.UZ = uz0.clone()
        self._scale_initial_axial_flow()
        ut_inlet = self.UT.clone()
        uz_inlet = self.UZ.clone()
        self._apply_initialization_boundary_to(self.UR, self.UT, self.UZ, ut_inlet, uz_inlet)

        if method_name in {"hybrid", "fluent_hybrid", "fluent"}:
            steps = self.initialization_sweeps if sweeps is None else max(int(sweeps), 0)
            phi_e, phi_w, phi_n, phi_s, phi_t, phi_b = self._face_opening()
            weights = torch.clamp(phi_e + phi_w + phi_n + phi_s + phi_t + phi_b, min=1e-6)

            def smooth_once(field: torch.Tensor, relax: float) -> torch.Tensor:
                acc = phi_e * self.neighbor(field, "E")
                acc = acc + phi_w * self.neighbor(field, "W")
                acc = acc + phi_n * self.neighbor(field, "N")
                acc = acc + phi_s * self.neighbor(field, "S")
                acc = acc + phi_t * self.neighbor(field, "T")
                acc = acc + phi_b * self.neighbor(field, "B")
                return field + relax * (acc / weights - field)

            for _ in range(steps):
                self.UR = smooth_once(self.UR, 0.42)
                self.UT = smooth_once(self.UT, 0.38)
                self.UZ = smooth_once(self.UZ, 0.45)
                self._apply_initialization_boundary_to(self.UR, self.UT, self.UZ, ut_inlet, uz_inlet)
            self._scale_initial_axial_flow()

        swirl_hat = self.UT if self.absolute_frame else self.UT + self._solid_ut()
        radial_pressure = 0.06 * (swirl_hat**2 - torch.mean(swirl_hat**2))
        blockage_pressure = 0.04 * (1.0 - torch.clamp(self.phi, min=0.0, max=1.0)) * (
            torch.sin(np.pi * torch.clamp(self.Z, min=0.0, max=1.0)) ** 2
        )
        self.P = self.delta_p_global * (1.0 - self.Z) + radial_pressure + blockage_pressure
        self._set_pressure_boundary()
        self._apply_velocity_boundary_to(self.UR, self.UT, self.UZ)
        self._apply_ibm()
        self._project_theta_periodic(self.UR, self.UT, self.UZ, self.P)
        if hasattr(self, "P_prime"):
            self.P_prime = torch.zeros_like(self.P)
        self.UR_tilde = self.UR.clone()
        self.UT_tilde = self.UT.clone()
        self.UZ_tilde = self.UZ.clone()
        return self.initialization_summary()

    def _initialize_physical_startup(self) -> None:
        self.initialize_flow(self.initialization, self.initialization_sweeps)

    def initialization_summary(self) -> dict[str, float]:
        return {
            "UR_std": float(torch.std(self.UR).item()),
            "UT_std": float(torch.std(self.UT).item()),
            "UZ_std": float(torch.std(self.UZ).item()),
            "P_std": float(torch.std(self.P).item()),
            "q_hat": self.outlet_flow_rate_hat(),
        }

    def set_blade_boundary(self, boundary_data: Any, band_cells: float = 1.5) -> None:
        def read_field(name: str) -> Any:
            if isinstance(boundary_data, dict):
                return boundary_data[name]
            return getattr(boundary_data, name)

        self.blade_mask = torch.as_tensor(read_field("mask"), dtype=torch.bool, device=self.device)
        self.blade_distance = torch.as_tensor(read_field("signed_distance"), dtype=torch.float32, device=self.device)
        self.blade_distance_z = torch.as_tensor(read_field("signed_distance_z"), dtype=torch.float32, device=self.device)
        self.blade_footprint = torch.as_tensor(read_field("footprint_mask"), dtype=torch.bool, device=self.device)
        self.blade_boundary_meta = read_field("metadata")
        self.blade_boundary_band_cells = float(band_cells)
        if hasattr(self, "P"):
            if self.adaptive_ibm:
                self.ibm_C, self.ibm_epsilon = self._ibm_target_parameters()
            self._refresh_ibm_phi()

    def attach_blade_boundary(self, blade_params_path: str | Path = "blade_params.json", band_cells: float = 1.5):
        return attach_blade_to_solver(self, blade_params_path, band_cells=band_cells)

    def _solid_ut(self) -> torch.Tensor:
        # absolute_frame=True 时，固体区周向速度按叶轮固体转动速度写入。
        if not self.absolute_frame:
            return torch.zeros_like(self.r_hatC)
        return self.sgn_omega * self.delta * self.r_hatC

    def _hard_solid_mask(self) -> torch.Tensor:
        return self.phi <= self.ibm_hard_phi

    def neighbor_raw(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        if direction == "E":
            out = torch.empty_like(x)
            out[:-1, :, :] = x[1:, :, :]
            out[-1, :, :] = x[-1, :, :]
            return out
        if direction == "W":
            out = torch.empty_like(x)
            out[1:, :, :] = x[:-1, :, :]
            out[0, :, :] = x[0, :, :]
            return out
        if direction == "N":
            out = torch.empty_like(x)
            out[:, :-1, :] = x[:, 1:, :]
            out[:, -1, :] = x[:, 1, :]
            return out
        if direction == "S":
            out = torch.empty_like(x)
            out[:, 1:, :] = x[:, :-1, :]
            out[:, 0, :] = x[:, -2, :]
            return out
        if direction == "T":
            out = torch.empty_like(x)
            out[:, :, :-1] = x[:, :, 1:]
            out[:, :, -1] = x[:, :, -1]
            return out
        if direction == "B":
            out = torch.empty_like(x)
            out[:, :, 1:] = x[:, :, :-1]
            out[:, :, 0] = x[:, :, 0]
            return out
        raise ValueError(f"unknown direction {direction!r}")

    def neighbor(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        return self.neighbor_raw(x, direction)

    def _project_theta_periodic(self, *fields: torch.Tensor) -> None:
        for field in fields:
            field[:, -1, :] = field[:, 0, :]

    @staticmethod
    def _normalize_flow_boundary_mode(mode: str) -> str:
        value = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "": "pressure",
            "none": "pressure",
            "off": "pressure",
            "pressure": "pressure",
            "pressure_drop": "pressure",
            "pressure_driven": "pressure",
            "zero_gradient": "pressure",
            "outlet": "outlet_flow",
            "outflow": "outlet_flow",
            "outlet_flow": "outlet_flow",
            "fixed_outlet_flow": "outlet_flow",
            "mass_flow_outlet": "outlet_flow",
            "inlet": "inlet_flow",
            "inflow": "inlet_flow",
            "inlet_flow": "inlet_flow",
            "fixed_inlet_flow": "inlet_flow",
            "mass_flow_inlet": "inlet_flow",
            "flow": "both_flow",
            "fixed_flow": "both_flow",
            "mass_flow": "both_flow",
            "balanced_flow": "both_flow",
            "inlet_outlet_flow": "both_flow",
            "both_flow": "both_flow",
        }
        if value not in aliases:
            valid = ", ".join(sorted({*aliases.values(), "pressure_driven", "zero_gradient"}))
            raise ValueError(f"Unknown flow_boundary_mode={mode!r}. Expected one of: {valid}.")
        return aliases[value]

    def _flow_boundary_sides(self) -> tuple[bool, bool]:
        inlet = self.flow_boundary_mode in {"inlet_flow", "both_flow"}
        outlet = self.flow_boundary_mode in {"outlet_flow", "both_flow"}
        if outlet and self.balance_inlet_flow:
            inlet = True
        return inlet, outlet

    def _boundary_fluid_opening(self, z_index: int) -> torch.Tensor:
        opening = torch.clamp(self.phi[:, :, z_index], min=0.0, max=1.0)
        hard = self._hard_solid_mask()[:, :, z_index]
        return torch.where(hard, torch.zeros_like(opening), opening)

    def _axial_boundary_raw_flow(self, uz: torch.Tensor, z_index: int) -> torch.Tensor:
        opening = self._boundary_fluid_opening(z_index)
        return torch.sum(uz[:, :, z_index] * opening * self.outlet_area_weight)

    def _target_axial_boundary_profile(self, z_index: int, target_raw_flow: torch.Tensor | None = None) -> torch.Tensor:
        opening = self._boundary_fluid_opening(z_index)
        area = self.outlet_area_weight
        open_area = torch.sum(opening * area)
        if (not bool(torch.isfinite(open_area).item())) or float(open_area.item()) <= 1e-12:
            return torch.zeros_like(opening)

        r_core = torch.clamp(4.0 * self.R[:, :, z_index] * (1.0 - self.R[:, :, z_index]), min=0.0, max=1.0)
        wall_shape = torch.sqrt(r_core)
        floor = min(self.flow_boundary_profile_floor, 0.95)
        profile = torch.clamp(wall_shape * torch.clamp(opening, min=floor), min=floor)
        profile[0, :] = 0.0
        profile[-1, :] = 0.0
        profile[:, -1] = profile[:, 0]

        raw_shape_flow = torch.sum(profile * opening * area)
        if (not bool(torch.isfinite(raw_shape_flow).item())) or float(raw_shape_flow.item()) <= 1e-12:
            return torch.zeros_like(opening)
        target = open_area if target_raw_flow is None else torch.as_tensor(target_raw_flow, dtype=profile.dtype, device=profile.device)
        if (not bool(torch.isfinite(target).item())) or float(target.item()) <= 0.0:
            return torch.zeros_like(opening)
        return profile * (target / torch.clamp(raw_shape_flow, min=1e-12))

    def _apply_flow_boundary_to(self, uz: torch.Tensor) -> None:
        inlet, outlet = self._flow_boundary_sides()
        if outlet:
            uz[:, :, -1] = self._target_axial_boundary_profile(-1)
        if inlet:
            target = self._axial_boundary_raw_flow(uz, -1) if outlet and self.balance_inlet_flow else None
            uz[:, :, 0] = self._target_axial_boundary_profile(0, target)

    def _set_pressure_boundary(self) -> None:
        self.P[:, :, 0] = self.delta_p_global
        self.P[:, :, -1] = 0.0
        self._project_theta_periodic(self.P)

    def _apply_velocity_boundary_to(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
    ) -> None:
        """把速度边界条件投影到给定三速度场上。

        这个 helper 同时用于主速度和 momentum predictor 速度。SIMPLE/COUPLE 中
        predictor 如果没有边界投影，会把边界噪声带进 Rhie-Chow 和压力修正方程。
        """
        # Inlet/outlet: pressure is fixed; selected flow sides override zero-gradient UZ.
        if self.n > 2:
            for field in (ur, ut, uz):
                field[:, :, 0] = field[:, :, 1]
                field[:, :, -1] = field[:, :, -2]
        self._apply_flow_boundary_to(uz)

        solid_ut = self._solid_ut()
        # Hub/shroud: no penetration and no axial slip in this first stable generator.
        ur[0, :, :] = 0.0
        ur[-1, :, :] = 0.0
        ut[0, :, :] = solid_ut[0, :, :]
        ut[-1, :, :] = solid_ut[-1, :, :]
        uz[0, :, :] = 0.0
        uz[-1, :, :] = 0.0
        self._project_theta_periodic(ur, ut, uz)

    def _set_velocity_boundary(self) -> None:
        self._apply_velocity_boundary_to(self.UR, self.UT, self.UZ)

    def _apply_ibm(self) -> None:
        hard = self._hard_solid_mask()
        solid_ut = self._solid_ut()
        self.UR = torch.where(hard, torch.zeros_like(self.UR), self.UR)
        self.UT = torch.where(hard, solid_ut, self.UT)
        self.UZ = torch.where(hard, torch.zeros_like(self.UZ), self.UZ)
        if hasattr(self, "UR_tilde"):
            self.UR_tilde = torch.where(hard, torch.zeros_like(self.UR_tilde), self.UR_tilde)
            self.UT_tilde = torch.where(hard, solid_ut, self.UT_tilde)
            self.UZ_tilde = torch.where(hard, torch.zeros_like(self.UZ_tilde), self.UZ_tilde)

    def _apply_boundary(self) -> None:
        self._set_pressure_boundary()
        self._set_velocity_boundary()
        self._apply_ibm()
        self._project_theta_periodic(self.UR, self.UT, self.UZ, self.P)

    def _pressure_gradient_r(self, p: torch.Tensor) -> torch.Tensor:
        grad = (self.neighbor(p, "E") - self.neighbor(p, "W")) / (2.0 * self.dR)
        grad[0, :, :] = (p[1, :, :] - p[0, :, :]) / self.dR
        grad[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / self.dR
        return grad

    def _pressure_gradient_theta(self, p: torch.Tensor) -> torch.Tensor:
        return (self.neighbor(p, "N") - self.neighbor(p, "S")) / (2.0 * self.dTheta)

    def _pressure_gradient_z(self, p: torch.Tensor) -> torch.Tensor:
        grad = (self.neighbor(p, "T") - self.neighbor(p, "B")) / (2.0 * self.dZ)
        grad[:, :, 0] = (p[:, :, 1] - p[:, :, 0]) / self.dZ
        grad[:, :, -1] = (p[:, :, -1] - p[:, :, -2]) / self.dZ
        return grad

    def _effective_inv_reynolds(self) -> torch.Tensor:
        inv_re = 1.0 / max(self.Re_omega, 1e-12)
        if self.rans_model in {"none", "off", "laminar"} or self.rans_mixing_length <= 0.0:
            self.nut_ratio = torch.zeros_like(self.P)
            return torch.full_like(self.P, inv_re)

        # 调试友好的零方程 RANS：用三速度分量梯度构造一个标量应变率。
        grads = [
            self._pressure_gradient_r(self.UR),
            self.K_theta_C * self._pressure_gradient_theta(self.UR),
            self.Lambda * self._pressure_gradient_z(self.UR),
            self._pressure_gradient_r(self.UT),
            self.K_theta_C * self._pressure_gradient_theta(self.UT),
            self.Lambda * self._pressure_gradient_z(self.UT),
            self._pressure_gradient_r(self.UZ),
            self.K_theta_C * self._pressure_gradient_theta(self.UZ),
            self.Lambda * self._pressure_gradient_z(self.UZ),
        ]
        strain = torch.sqrt(sum(g * g for g in grads) + 1e-20)
        mixing_length = torch.full_like(strain, self.rans_mixing_length)
        if self.blade_distance is not None:
            wall_distance = torch.clamp(self.blade_distance, min=0.0)
            mixing_length = torch.minimum(mixing_length, 0.41 * wall_distance + min(self.dR, self.dTheta, self.dZ))
        nut_ratio = (mixing_length**2) * strain * self.Re_omega * torch.clamp(self.phi, 0.0, 1.0) ** 2
        nut_ratio = torch.clamp(nut_ratio, min=0.0, max=self.rans_nut_max_ratio)
        for _ in range(max(int(self.rans_smoothing_steps), 0)):
            nut_ratio = 0.5 * nut_ratio + (1.0 / 12.0) * sum(
                self.neighbor(nut_ratio, d) for d in ("E", "W", "N", "S", "T", "B")
            )
            nut_ratio = torch.clamp(nut_ratio, min=0.0, max=self.rans_nut_max_ratio)
            self._project_theta_periodic(nut_ratio)
        self.nut_ratio = nut_ratio
        return inv_re * (1.0 + nut_ratio)

    def rhie_chow(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        p: torch.Tensor,
        a_r: torch.Tensor,
        a_t: torch.Tensor,
        a_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """三维 Rhie-Chow 面速度。

        对应 PDF 式 (2.54)-(2.57) 与式 (2.93)。这里使用严格的“面梯度减去单元
        中心梯度平均”的结构，防止同位网格压力棋盘振荡。
        """
        a_r = a_r + (1.0 - self.phi) + 1e-6
        a_t = a_t + (1.0 - self.phi) + 1e-6
        a_z = a_z + (1.0 - self.phi) + 1e-6

        gr_c = self.dV * self.Eu_omega * self._pressure_gradient_r(p)
        gt_c = self.dV * self.Eu_omega * self.K_theta_C * self._pressure_gradient_theta(p)
        gz_c = self.dV * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * self._pressure_gradient_z(p)

        # 这里走rhie-chow插值的思想
        # Eq. (2.93): face velocity with the full local A^-1 pressure-gradient block.
        a12 = torch.nan_to_num(getattr(self, "A12", torch.zeros_like(a_r)), nan=0.0, posinf=0.0, neginf=0.0)
        a21 = torch.nan_to_num(getattr(self, "A21", torch.zeros_like(a_r)), nan=0.0, posinf=0.0, neginf=0.0)
        det2 = torch.clamp(a_r * a_t - a12 * a21, min=1e-12)
        inv11 = a_t / det2
        inv12 = -a12 / det2
        inv21 = -a21 / det2
        inv22 = a_r / det2
        inv33 = 1.0 / torch.clamp(a_z, min=1e-12)
        qr_c = inv11 * gr_c + inv12 * gt_c
        qt_c = inv21 * gr_c + inv22 * gt_c
        qz_c = inv33 * gz_c

        pe, pw = self.neighbor(p, "E"), self.neighbor(p, "W")
        pn, ps = self.neighbor(p, "N"), self.neighbor(p, "S")
        pne = self.neighbor(pe, "N")
        pnw = self.neighbor(pw, "N")
        pse = self.neighbor(pe, "S")
        psw = self.neighbor(pw, "S")

        def avg_face(x: torch.Tensor, direction: str) -> torch.Tensor:
            return 0.5 * (x + self.neighbor(x, direction))

        k_e = avg_face(self.K_theta_C, "E")
        k_w = avg_face(self.K_theta_C, "W")
        qr_e = avg_face(inv11, "E") * (self.dV * self.Eu_omega * (pe - p) / self.dR)
        qr_e = qr_e + avg_face(inv12, "E") * (
            self.dV * self.Eu_omega * k_e * (pn + pne - ps - pse) / (4.0 * self.dTheta)
        )
        qr_w = avg_face(inv11, "W") * (self.dV * self.Eu_omega * (p - pw) / self.dR)
        qr_w = qr_w + avg_face(inv12, "W") * (
            self.dV * self.Eu_omega * k_w * (pn + pnw - ps - psw) / (4.0 * self.dTheta)
        )
        qt_n = avg_face(inv21, "N") * (self.dV * self.Eu_omega * (pe + pne - pw - pnw) / (4.0 * self.dR))
        qt_n = qt_n + avg_face(inv22, "N") * (self.dV * self.Eu_omega * self.K_theta_C * (pn - p) / self.dTheta)
        qt_s = avg_face(inv21, "S") * (self.dV * self.Eu_omega * (pe + pse - pw - psw) / (4.0 * self.dR))
        qt_s = qt_s + avg_face(inv22, "S") * (self.dV * self.Eu_omega * self.K_theta_C * (p - ps) / self.dTheta)
        qz_t = avg_face(inv33, "T") * (
            self.dV * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * (self.neighbor(p, "T") - p) / self.dZ
        )
        qz_b = avg_face(inv33, "B") * (
            self.dV * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * (p - self.neighbor(p, "B")) / self.dZ
        )

        ur_e = 0.5 * (ur + self.neighbor(ur, "E")) - (qr_e - 0.5 * (qr_c + self.neighbor(qr_c, "E")))
        ur_w = 0.5 * (ur + self.neighbor(ur, "W")) - (qr_w - 0.5 * (qr_c + self.neighbor(qr_c, "W")))
        ut_n = 0.5 * (ut + self.neighbor(ut, "N")) - (qt_n - 0.5 * (qt_c + self.neighbor(qt_c, "N")))
        ut_s = 0.5 * (ut + self.neighbor(ut, "S")) - (qt_s - 0.5 * (qt_c + self.neighbor(qt_c, "S")))
        uz_t = 0.5 * (uz + self.neighbor(uz, "T")) - (qz_t - 0.5 * (qz_c + self.neighbor(qz_c, "T")))
        uz_b = 0.5 * (uz + self.neighbor(uz, "B")) - (qz_b - 0.5 * (qz_c + self.neighbor(qz_c, "B")))

        ur_e[-1, :, :] = 0.0
        ur_w[0, :, :] = 0.0
        inlet_flow, outlet_flow = self._flow_boundary_sides()
        inlet_target = None
        if outlet_flow:
            outlet_profile = self._target_axial_boundary_profile(-1)
            uz_t[:, :, -1] = outlet_profile
            if inlet_flow and self.balance_inlet_flow:
                inlet_target = torch.sum(outlet_profile * self._boundary_fluid_opening(-1) * self.outlet_area_weight)
        else:
            uz_t[:, :, -1] = uz[:, :, -1]
        uz_b[:, :, 0] = self._target_axial_boundary_profile(0, inlet_target) if inlet_flow else uz[:, :, 0]
        self._project_theta_periodic(ur_e, ur_w, ut_n, ut_s, uz_t, uz_b)
        return ur_e, ur_w, ut_n, ut_s, uz_t, uz_b

    def _face_fluxes(
        self,
        ur_e: torch.Tensor,
        ur_w: torch.Tensor,
        ut_n: torch.Tensor,
        ut_s: torch.Tensor,
        uz_t: torch.Tensor,
        uz_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fe = (self.r_hat_E / torch.clamp(self.r_hatC, min=1e-12)) * self.radial_area * ur_e
        fw = -(self.r_hat_W / torch.clamp(self.r_hatC, min=1e-12)) * self.radial_area * ur_w
        fn = self.K_theta_C * self.theta_area * ut_n
        fs = -self.K_theta_C * self.theta_area * ut_s
        ft = self.Lambda * self.Ku * self.z_area * uz_t
        fb = -self.Lambda * self.Ku * self.z_area * uz_b
        return fe, fw, fn, fs, ft, fb

    def _face_opening(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_e = torch.minimum(self.phi, self.neighbor(self.phi, "E"))
        phi_w = torch.minimum(self.phi, self.neighbor(self.phi, "W"))
        phi_n = torch.minimum(self.phi, self.neighbor(self.phi, "N"))
        phi_s = torch.minimum(self.phi, self.neighbor(self.phi, "S"))
        phi_t = torch.minimum(self.phi, self.neighbor(self.phi, "T"))
        phi_b = torch.minimum(self.phi, self.neighbor(self.phi, "B"))
        phi_e[-1, :, :] = 0.0
        phi_w[0, :, :] = 0.0
        phi_t[:, :, -1] = self.phi[:, :, -1]
        phi_b[:, :, 0] = self.phi[:, :, 0]
        self._project_theta_periodic(phi_e, phi_w, phi_n, phi_s, phi_t, phi_b)
        return phi_e, phi_w, phi_n, phi_s, phi_t, phi_b

    def _assemble_momentum_coefficients(self) -> dict[str, torch.Tensor]:
        ur_e, ur_w, ut_n, ut_s, uz_t, uz_b = self.rhie_chow(
            self.UR,
            self.UT,
            self.UZ,
            self.P,
            self.A11,
            self.A22,
            self.A33,
        )
        fe, fw, fn, fs, ft, fb = self._face_fluxes(ur_e, ur_w, ut_n, ut_s, uz_t, uz_b)
        phi_e, phi_w, phi_n, phi_s, phi_t, phi_b = self._face_opening()
        fe, fw, fn, fs, ft, fb = fe * phi_e, fw * phi_w, fn * phi_n, fs * phi_s, ft * phi_t, fb * phi_b

        inv_re = self._effective_inv_reynolds()
        inv_re_e = 0.5 * (inv_re + self.neighbor(inv_re, "E"))
        inv_re_w = 0.5 * (inv_re + self.neighbor(inv_re, "W"))
        inv_re_n = 0.5 * (inv_re + self.neighbor(inv_re, "N"))
        inv_re_s = 0.5 * (inv_re + self.neighbor(inv_re, "S"))
        inv_re_t = 0.5 * (inv_re + self.neighbor(inv_re, "T"))
        inv_re_b = 0.5 * (inv_re + self.neighbor(inv_re, "B"))

        de = phi_e * (self.r_hat_E / torch.clamp(self.r_hatC, min=1e-12)) * self.radial_area * inv_re_e / self.dR
        dw = phi_w * (self.r_hat_W / torch.clamp(self.r_hatC, min=1e-12)) * self.radial_area * inv_re_w / self.dR
        dn = phi_n * (self.K_theta_C**2) * self.theta_area * inv_re_n / self.dTheta
        ds = phi_s * (self.K_theta_C**2) * self.theta_area * inv_re_s / self.dTheta
        dt = phi_t * (self.Lambda**2) * self.z_area * inv_re_t / self.dZ
        db = phi_b * (self.Lambda**2) * self.z_area * inv_re_b / self.dZ

        aE = de - torch.clamp(fe, max=0.0)
        aW = dw - torch.clamp(fw, max=0.0)
        aN = dn - torch.clamp(fn, max=0.0)
        aS = ds - torch.clamp(fs, max=0.0)
        aT = dt - torch.clamp(ft, max=0.0)
        aB = db - torch.clamp(fb, max=0.0)
        aP_conv = sum(torch.clamp(f, min=0.0) for f in (fe, fw, fn, fs, ft, fb))
        aP_diff = de + dw + dn + ds + dt + db

        solid = 1.0 - self.phi
        pseudo = self.dV / self.pseudo_dt
        if self.local_pseudo_dt:
            spectral_radius = sum(torch.abs(f) for f in (fe, fw, fn, fs, ft, fb)) + 2.0 * aP_diff + solid
            pseudo = pseudo + spectral_radius / self.pseudo_cfl

        base = aP_conv + aP_diff + solid + pseudo + 1e-12
        geom = self.dV * inv_re / torch.clamp(self.r_hatC**2, min=1e-12)
        A11 = base + geom
        coupling_limit = 2.0
        # PDF (2.79)-(2.85): keep the R-Theta local block coupled through A12/A21.
        ur_clip = torch.clamp(self.UR, min=-coupling_limit, max=coupling_limit)
        ut_clip = torch.clamp(self.UT, min=-coupling_limit, max=coupling_limit)
        rhat = torch.clamp(self.r_hatC, min=1e-12)
        A22 = base + geom + self.cross_coupling_strength * self.dV * ur_clip / rhat
        A22 = torch.clamp(A22, min=0.25 * (base + geom) + 1e-12)
        A33 = base

        # A12/A21 是 R-Theta 动量局部耦合项。默认强度为 0，保证初版稳定；
        # 若要检查 PDF 中完整旋转平面局部块，可逐步增大 cross_coupling_strength。
        coupling_limit = 2.0
        ur_clip = torch.clamp(self.UR, min=-coupling_limit, max=coupling_limit)
        ut_clip = torch.clamp(self.UT, min=-coupling_limit, max=coupling_limit)
        A12 = -2.0 * self.cross_coupling_strength * self.dV * ut_clip / rhat
        A21 = self.cross_coupling_strength * self.dV * ut_clip / rhat
        if not self.absolute_frame:
            A12 = A12 + 2.0 * self.sgn_omega * self.delta * self.dV
            A21 = A21 - 2.0 * self.sgn_omega * self.delta * self.dV
        det_limit = self.coupling_det_fraction * torch.clamp(A11 * A22, min=1e-24)
        prod_abs = torch.abs(A12 * A21)
        scale = torch.sqrt(torch.clamp(det_limit / torch.clamp(prod_abs, min=1e-24), max=1.0))
        A12 = A12 * scale
        A21 = A21 * scale

        return {
            "E": aE,
            "W": aW,
            "N": aN,
            "S": aS,
            "T": aT,
            "B": aB,
            "A11": A11,
            "A12": A12,
            "A21": A21,
            "A22": A22,
            "A33": A33,
            "pseudo": pseudo,
            "inv_re": inv_re,
        }

    def momentum(self) -> None:
        coef = self._assemble_momentum_coefficients()
        self.A11, self.A12, self.A21 = coef["A11"], coef["A12"], coef["A21"]
        self.A22, self.A33 = coef["A22"], coef["A33"]

        p_r = self.Eu_omega * self._pressure_gradient_r(self.P)
        p_t = self.Eu_omega * self.K_theta_C * self._pressure_gradient_theta(self.P)
        p_z = self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * self._pressure_gradient_z(self.P)

        UR_ref, UT_ref, UZ_ref = self.UR.clone(), self.UT.clone(), self.UZ.clone()
        UR_iter, UT_iter, UZ_iter = self.UR.clone(), self.UT.clone(), self.UZ.clone()
        hard = self._hard_solid_mask()
        solid = 1.0 - self.phi
        solid_ut = self._solid_ut()
        relax = float(np.clip(self.momentum_solver_relax, 0.05, 1.0))

        for _ in range(max(int(self.momentum_sweeps), 1)):
            rhs_r = (
                coef["E"] * self.neighbor(UR_iter, "E")
                + coef["W"] * self.neighbor(UR_iter, "W")
                + coef["N"] * self.neighbor(UR_iter, "N")
                + coef["S"] * self.neighbor(UR_iter, "S")
                + coef["T"] * self.neighbor(UR_iter, "T")
                + coef["B"] * self.neighbor(UR_iter, "B")
                + coef["pseudo"] * UR_ref
                - self.dV * p_r
            )
            rhs_t = (
                coef["E"] * self.neighbor(UT_iter, "E")
                + coef["W"] * self.neighbor(UT_iter, "W")
                + coef["N"] * self.neighbor(UT_iter, "N")
                + coef["S"] * self.neighbor(UT_iter, "S")
                + coef["T"] * self.neighbor(UT_iter, "T")
                + coef["B"] * self.neighbor(UT_iter, "B")
                + coef["pseudo"] * UT_ref
                + solid * solid_ut
                - self.dV * p_t
            )
            rhs_z = (
                coef["E"] * self.neighbor(UZ_iter, "E")
                + coef["W"] * self.neighbor(UZ_iter, "W")
                + coef["N"] * self.neighbor(UZ_iter, "N")
                + coef["S"] * self.neighbor(UZ_iter, "S")
                + coef["T"] * self.neighbor(UZ_iter, "T")
                + coef["B"] * self.neighbor(UZ_iter, "B")
                + coef["pseudo"] * UZ_ref
                - self.dV * p_z
                - self.dV * self.G_star
            )

            # 旋转源项只作为显式、限幅、欠松弛贡献，避免初版 3D 生成器被几何源项拖散。
            if self.cross_coupling_strength > 0.0:
                rhat = torch.clamp(self.r_hatC, min=1e-12)
                inv_re = coef["inv_re"]
                ut_star = torch.clamp(UT_ref, -2.0, 2.0)
                ur_star = torch.clamp(UR_ref, -2.0, 2.0)
                cross_diff_r = self.K_theta_C * inv_re * (self.neighbor(UT_iter, "N") - self.neighbor(UT_iter, "S")) / (self.dTheta * rhat)
                cross_diff_t = self.K_theta_C * inv_re * (self.neighbor(UR_iter, "N") - self.neighbor(UR_iter, "S")) / (self.dTheta * rhat)
                rhs_r = rhs_r - self.cross_coupling_strength * self.dV * (ut_star**2 / rhat + cross_diff_r)
                rhs_t = rhs_t + self.cross_coupling_strength * self.dV * (ur_star * ut_star / rhat + cross_diff_t)
                if not self.absolute_frame:
                    rhs_r = rhs_r + self.dV * (self.delta**2) * rhat

            det2 = torch.clamp(self.A11 * self.A22 - self.A12 * self.A21, min=1e-12)
            UR_new = (rhs_r * self.A22 - self.A12 * rhs_t) / det2
            UT_new = (self.A11 * rhs_t - self.A21 * rhs_r) / det2
            UZ_new = rhs_z / torch.clamp(self.A33, min=1e-12)

            UR_new = torch.where(hard, torch.zeros_like(UR_new), UR_new)
            UT_new = torch.where(hard, solid_ut, UT_new)
            UZ_new = torch.where(hard, torch.zeros_like(UZ_new), UZ_new)
            UR_iter = UR_iter + relax * (UR_new - UR_iter)
            UT_iter = UT_iter + relax * (UT_new - UT_iter)
            UZ_iter = UZ_iter + relax * (UZ_new - UZ_iter)
            self._apply_velocity_boundary_to(UR_iter, UT_iter, UZ_iter)

        self.UR_tilde, self.UT_tilde, self.UZ_tilde = UR_iter, UT_iter, UZ_iter
        self._project_theta_periodic(self.UR_tilde, self.UT_tilde, self.UZ_tilde)

    def pressure(self) -> None:
        self.pressure_updater.simple_step()

    def _rhie_flux_divergence(self, ur: torch.Tensor, ut: torch.Tensor, uz: torch.Tensor, p: torch.Tensor):
        faces = self.rhie_chow(ur, ut, uz, p, self.A11, self.A22, self.A33)
        fluxes = self._face_fluxes(*faces)
        openings = self._face_opening()
        return sum(f * o for f, o in zip(fluxes, openings))

    def _normalized_continuity_residual(self, div: torch.Tensor) -> torch.Tensor:
        return div / torch.clamp(self.dV, min=1e-12)

    def _momentum_residual(self, coef: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_r = self.Eu_omega * self._pressure_gradient_r(self.P)
        p_t = self.Eu_omega * self.K_theta_C * self._pressure_gradient_theta(self.P)
        p_z = self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * self._pressure_gradient_z(self.P)
        def neighbor_part(field: torch.Tensor) -> torch.Tensor:
            return (
                coef["E"] * self.neighbor(field, "E")
                + coef["W"] * self.neighbor(field, "W")
                + coef["N"] * self.neighbor(field, "N")
                + coef["S"] * self.neighbor(field, "S")
                + coef["T"] * self.neighbor(field, "T")
                + coef["B"] * self.neighbor(field, "B")
            )

        rhat = torch.clamp(self.r_hatC, min=1e-12)
        inv_re = coef.get("inv_re", self._effective_inv_reynolds())
        cross_diff_r = self.K_theta_C * inv_re * (self.neighbor(self.UT, "N") - self.neighbor(self.UT, "S")) / (self.dTheta * rhat)
        cross_diff_t = self.K_theta_C * inv_re * (self.neighbor(self.UR, "N") - self.neighbor(self.UR, "S")) / (self.dTheta * rhat)
        rr = (
            coef["A11"] * self.UR
            + coef["A12"] * self.UT
            - neighbor_part(self.UR)
            - coef["pseudo"] * self.UR
            + self.dV * p_r
        )
        rt = (
            coef["A21"] * self.UR
            + coef["A22"] * self.UT
            - neighbor_part(self.UT)
            - coef["pseudo"] * self.UT
            - (1.0 - self.phi) * self._solid_ut()
            + self.dV * p_t
        )
        rz = (
            coef["A33"] * self.UZ
            - neighbor_part(self.UZ)
            - coef["pseudo"] * self.UZ
            + self.dV * p_z
            + self.dV * self.G_star
        )
        if self.cross_coupling_strength > 0.0:
            rr = rr + self.cross_coupling_strength * self.dV * (self.UT**2 / rhat + cross_diff_r)
            rt = rt - self.cross_coupling_strength * self.dV * (self.UR * self.UT / rhat + cross_diff_t)
            if not self.absolute_frame:
                rr = rr - self.dV * (self.delta**2) * rhat
        return rr, rt, rz

    def _fluid_residual_mask(self) -> torch.Tensor:
        mask = self.phi > 0.2
        mask[0, :, :] = False
        mask[-1, :, :] = False
        mask[:, :, 0] = False
        mask[:, :, -1] = False
        return mask

    @staticmethod
    def _masked_absmax(field: torch.Tensor, mask: torch.Tensor) -> float:
        if torch.any(mask):
            return float(torch.max(torch.abs(field[mask])).item())
        return float(torch.max(torch.abs(field)).item())

    @staticmethod
    def _fields_are_finite(*fields: torch.Tensor) -> bool:
        return all(bool(torch.isfinite(field).all().item()) for field in fields)

    def _velocity_update_scale(
        self,
        d_ur: torch.Tensor,
        d_ut: torch.Tensor,
        d_uz: torch.Tensor,
        limit: float | None,
    ) -> float:
        if limit is None or limit <= 0.0:
            return 1.0
        mask = self.phi > 0.2
        values = [torch.max(torch.abs(x[mask])) if torch.any(mask) else torch.max(torch.abs(x)) for x in (d_ur, d_ut, d_uz)]
        max_value = float(torch.maximum(torch.maximum(values[0], values[1]), values[2]).item())
        if not np.isfinite(max_value):
            return 0.0
        if max_value <= limit:
            return 1.0
        return max(float(limit) / max(max_value, 1e-30), 0.0)

    def _pressure_update_scale(self, d_p: torch.Tensor, limit: float | None) -> float:
        if limit is None or limit <= 0.0:
            return 1.0
        max_update = float(torch.max(torch.abs(d_p)).item())
        if not np.isfinite(max_update):
            return 0.0
        if max_update <= limit:
            return 1.0
        return max(float(limit) / max(max_update, 1e-30), 0.0)

    def _clamp_couple_fields(self) -> None:
        limit = float(self.couple_field_abs_limit)
        if limit <= 0.0:
            return
        self.UR = torch.clamp(self.UR, min=-limit, max=limit)
        self.UT = torch.clamp(self.UT, min=-limit, max=limit)
        self.UZ = torch.clamp(self.UZ, min=-limit, max=limit)
        p_limit = max(limit, 4.0 * abs(self.delta_p_global), 1.0)
        self.P = torch.clamp(self.P, min=-p_limit, max=p_limit)
        self._apply_boundary()

    def momentum_error(self) -> float:
        coef = self._assemble_momentum_coefficients()
        rr, rt, rz = self._momentum_residual(coef)
        mask = self._fluid_residual_mask()
        return max(self._masked_absmax(rr, mask), self._masked_absmax(rt, mask), self._masked_absmax(rz, mask))

    def continuity_error_integrated(self) -> float:
        div = self._rhie_flux_divergence(self.UR, self.UT, self.UZ, self.P)
        mask = self._fluid_residual_mask()
        if torch.any(mask):
            return float(torch.max(torch.abs(div[mask])).item())
        return float(torch.max(torch.abs(div)).item())

    def continuity_error(self) -> float:
        div = self._rhie_flux_divergence(self.UR, self.UT, self.UZ, self.P)
        residual = self._normalized_continuity_residual(div)
        mask = self._fluid_residual_mask()
        if torch.any(mask):
            return float(torch.max(torch.abs(residual[mask])).item())
        return float(torch.max(torch.abs(residual)).item())

    def _axial_boundary_flow_rate_hat(self, z_index: int) -> float:
        fluid = self._boundary_fluid_opening(z_index)
        weighted = self.UZ[:, :, z_index] * fluid * self.outlet_area_weight
        q_raw = torch.sum(weighted)
        target = torch.sum(fluid * self.outlet_area_weight)
        return float((q_raw / torch.clamp(target, min=1e-12)).item())

    def inlet_flow_rate_hat(self) -> float:
        return self._axial_boundary_flow_rate_hat(0)

    def outlet_flow_rate_hat(self) -> float:
        # UZ is nondimensionalized by u_zo=qv/[pi(rs^2-rh^2)], so a weighted
        # boundary average of one matches the target passage flow.
        return self._axial_boundary_flow_rate_hat(-1)

    def _nudge_delta_p(self, q_hat: float, prev: tuple[float, float] | None) -> tuple[float, float]:
        err = q_hat - 1.0
        old_dp = self.delta_p_global
        if prev is not None:
            prev_dp, prev_err = prev
            denom = err - prev_err
            candidate = old_dp - err * (old_dp - prev_dp) / denom if abs(denom) > 1e-8 else old_dp * (1.0 - 0.5 * err)
        else:
            candidate = old_dp * (1.0 - 0.5 * err)
        if old_dp <= self.delta_p_min * 10.0 and err < 0.0:
            candidate = max(candidate, 0.02)
        if not np.isfinite(candidate) or candidate <= 0.0:
            candidate = old_dp * (1.25 if err < 0.0 else 0.75)
        candidate = float(np.clip(candidate, self.delta_p_min, self.delta_p_max))
        new_dp = (1.0 - self.delta_p_relax) * old_dp + self.delta_p_relax * candidate
        self.P = self.P + (new_dp - old_dp) * (1.0 - self.Z)
        self.delta_p_global = new_dp
        self._set_pressure_boundary()
        return old_dp, err

    def _relax_delta_p_to_flow(self, q_hat: float) -> None:
        if not np.isfinite(q_hat):
            return
        err = q_hat - 1.0
        old_dp = self.delta_p_global
        log_factor = float(np.clip(-self.couple_flow_control_gain * err, -self.couple_flow_control_clip, self.couple_flow_control_clip))
        candidate = old_dp * float(np.exp(log_factor))
        if old_dp <= self.delta_p_min * 10.0 and err < 0.0:
            candidate = max(candidate, 0.02)
        if not np.isfinite(candidate) or candidate <= 0.0:
            candidate = old_dp * (1.05 if err < 0.0 else 0.95)
        new_dp = float(np.clip(candidate, self.delta_p_min, self.delta_p_max))
        self.P = self.P + (new_dp - old_dp) * (1.0 - self.Z)
        self.delta_p_global = new_dp
        self._set_pressure_boundary()

    def simple_step(self) -> tuple[float, float, float]:
        old = (self.UR.clone(), self.UT.clone(), self.UZ.clone())
        self.momentum()
        self.pressure()
        self._apply_boundary()
        update = torch.maximum(
            torch.maximum(torch.max(torch.abs(self.UR - old[0])), torch.max(torch.abs(self.UT - old[1]))),
            torch.max(torch.abs(self.UZ - old[2])),
        )
        mass = self.continuity_error()
        momentum = self.momentum_error()
        self._adapt_ibm(mass, momentum)
        return float(update.item()), self.continuity_error(), self.momentum_error()

    def couple_step(self) -> tuple[float, float]:
        step_count = int(getattr(self, "_couple_step_count", 0))
        old = (self.UR.clone(), self.UT.clone(), self.UZ.clone(), self.P.clone(), self.P_prime.clone())
        self.momentum()

        d_ur = self.couple_relax * (self.UR_tilde - old[0])
        d_ut = self.couple_relax * (self.UT_tilde - old[1])
        d_uz = self.couple_relax * (self.UZ_tilde - old[2])
        scale = self._velocity_update_scale(d_ur, d_ut, d_uz, self.couple_momentum_update_limit)
        self.UR = old[0] + scale * d_ur
        self.UT = old[1] + scale * d_ut
        self.UZ = old[2] + scale * d_uz
        self._apply_boundary()
        self._clamp_couple_fields()

        if not self._fields_are_finite(self.UR, self.UT, self.UZ, self.P):
            self.UR, self.UT, self.UZ, self.P, self.P_prime = (x.clone() for x in old)
            self._apply_boundary()
            return 0.0, self.continuity_error()

        do_pressure = (
            self.couple_pressure_sweeps > 0
            and self.couple_pressure_interval > 0
            and step_count % self.couple_pressure_interval == 0
        )
        projection_info = {
            "attempted": float(do_pressure),
            "accepted": 0.0,
            "relax": 0.0,
            "base_mass": float("nan"),
            "new_mass": float("nan"),
            "base_momentum": float("nan"),
            "new_momentum": float("nan"),
        }
        for _ in range(self.couple_pressure_sweeps if do_pressure else 0):
            beta = -self._rhie_flux_divergence(self.UR, self.UT, self.UZ, self.P)
            base_mass = self.continuity_error()
            base_momentum = self.momentum_error()
            projection_info["base_mass"] = base_mass
            projection_info["base_momentum"] = base_momentum
            saved = (self.UR.clone(), self.UT.clone(), self.UZ.clone(), self.P.clone(), self.P_prime.clone())
            relax = self.pressure_projection_relax
            accepted = False
            while relax >= self.couple_pressure_min_relax:
                self.pressure_updater.project(
                    beta,
                    relax,
                    self.couple_pressure_velocity_limit,
                    self.couple_pressure_update_limit,
                )
                self._clamp_couple_fields()
                new_mass = self.continuity_error()
                new_momentum = self.momentum_error()
                projection_info["new_mass"] = new_mass
                projection_info["new_momentum"] = new_momentum
                finite = self._fields_are_finite(self.UR, self.UT, self.UZ, self.P)
                momentum_limit = max(
                    base_momentum * self.couple_pressure_max_momentum_growth,
                    base_momentum + self.couple_pressure_momentum_allowance,
                )
                mass_limit = base_mass
                if new_momentum > base_momentum:
                    drop = float(np.clip(self.couple_pressure_min_mass_drop_on_momentum_growth, 0.0, 0.95))
                    mass_limit = base_mass * (1.0 - drop)
                if finite and new_mass <= mass_limit and new_momentum <= momentum_limit:
                    accepted = True
                    projection_info["accepted"] = 1.0
                    projection_info["relax"] = relax
                    break
                self.UR, self.UT, self.UZ, self.P, self.P_prime = (x.clone() for x in saved)
                self._apply_boundary()
                if not self.couple_pressure_backtracking:
                    break
                relax *= 0.5
            if not accepted:
                self.UR, self.UT, self.UZ, self.P, self.P_prime = (x.clone() for x in saved)
                self._apply_boundary()
                break

        self.last_pressure_projection = projection_info
        self._couple_step_count = step_count + 1
        update = torch.maximum(
            torch.maximum(torch.max(torch.abs(self.UR - old[0])), torch.max(torch.abs(self.UT - old[1]))),
            torch.max(torch.abs(self.UZ - old[2])),
        )
        mass = self.continuity_error()
        self._adapt_ibm(mass, None)
        return float(update.item()), self.continuity_error()

    def solve(
        self,
        max_outer: int = 8,
        inner_per_outer: int | None = None,
        report_interval: int = 25,
        method: str = "simple",
    ) -> SolveLog3D:
        method = method.lower()
        if method == "couple":
            return self.solve_couple(max_iter=self.max_iter, report_interval=report_interval)
        self.last_solve_method = "simple"
        inner_limit = int(inner_per_outer or max(1, self.max_iter // max(max_outer, 1)))
        prev_secant: tuple[float, float] | None = None
        history: list[dict[str, float]] = []
        iteration_history: list[dict[str, float]] = []
        total_iter = 0
        converged = False
        final_update = final_mass = final_momentum = float("inf")

        for outer in range(max_outer):
            for inner in range(inner_limit):
                final_update, final_mass, final_momentum = self.simple_step()
                total_iter += 1
                q_hat = self.outlet_flow_rate_hat()
                iteration_history.append(
                    {
                        "iteration": float(total_iter),
                        "outer": float(outer),
                        "inner": float(inner),
                        "momentum": final_momentum,
                        "continuity": final_mass,
                        "update": final_update,
                        "q_hat": q_hat,
                        "delta_p": self.delta_p_global,
                        "nut_max": float(torch.max(self.nut_ratio).item()),
                        "nut_mean": float(torch.mean(self.nut_ratio).item()),
                    }
                )
                if report_interval and (inner % report_interval == 0 or inner == inner_limit - 1):
                    print(
                        f"SIMPLE3D outer={outer:02d} inner={inner:04d} "
                        f"mom={final_momentum:.3e} mass={final_mass:.3e} "
                        f"update={final_update:.3e} q_hat={q_hat:.5f} dP={self.delta_p_global:.5e}"
                    )
                if final_update < self.tol and final_mass < 5.0 * self.tol and final_momentum < 5.0 * self.tol:
                    break

            q_hat = self.outlet_flow_rate_hat()
            q_err = q_hat - 1.0
            history.append(
                {
                    "outer": float(outer),
                    "iterations": float(total_iter),
                    "update": final_update,
                    "momentum": final_momentum,
                    "mass": final_mass,
                    "q_hat": q_hat,
                    "delta_p": self.delta_p_global,
                }
            )
            if abs(q_err) < self.outer_flow_tol and final_mass < 1e-3 and final_momentum < 1e-3:
                converged = True
                break
            prev_secant = self._nudge_delta_p(q_hat, prev_secant)

        self.last_history = history
        self.iteration_history = iteration_history
        return SolveLog3D(
            method="SIMPLE3D",
            converged=converged,
            iterations=total_iter,
            final_momentum=final_momentum,
            final_mass=final_mass,
            final_update=final_update,
            final_q_hat=self.outlet_flow_rate_hat(),
            delta_p=self.delta_p_global,
        )

    def solve_couple(self, max_iter: int | None = None, report_interval: int = 25) -> SolveLog3D:
        limit = int(max_iter or self.max_iter)
        converged = False
        final_update = final_mass = final_momentum = float("inf")
        self.last_solve_method = "couple"
        self.iteration_history = []
        self._couple_step_count = 0
        total_iter = 0
        for it in range(limit):
            final_update, final_mass = self.couple_step()
            total_iter = it + 1
            final_momentum = self.momentum_error()
            q_hat = self.outlet_flow_rate_hat()
            projection = getattr(self, "last_pressure_projection", {})
            self.iteration_history.append(
                {
                    "iteration": float(it + 1),
                    "outer": 0.0,
                    "inner": float(it),
                    "momentum": final_momentum,
                    "continuity": final_mass,
                    "update": final_update,
                    "q_hat": q_hat,
                    "delta_p": self.delta_p_global,
                    "nut_max": float(torch.max(self.nut_ratio).item()),
                    "nut_mean": float(torch.mean(self.nut_ratio).item()),
                    "pressure_projection_attempted": float(projection.get("attempted", 0.0)),
                    "pressure_projection_accepted": float(projection.get("accepted", 0.0)),
                    "pressure_projection_relax": float(projection.get("relax", 0.0)),
                }
            )
            if report_interval and (it % report_interval == 0 or it == limit - 1):
                print(
                    f"COUPLE3D iter={it:04d} mom={final_momentum:.3e} "
                    f"mass={final_mass:.3e} update={final_update:.3e} "
                    f"q_hat={q_hat:.5f} dP={self.delta_p_global:.5e}"
                )
            if self.couple_flow_control and self.couple_flow_control_interval > 0 and (it + 1) % self.couple_flow_control_interval == 0:
                self._relax_delta_p_to_flow(q_hat)
            if final_update < self.tol and final_mass < 5.0 * self.tol and final_momentum < 5.0 * self.tol:
                converged = True
                break
        return SolveLog3D(
            method="COUPLE3D",
            converged=converged,
            iterations=total_iter,
            final_momentum=final_momentum,
            final_mass=final_mass,
            final_update=final_update,
            final_q_hat=self.outlet_flow_rate_hat(),
            delta_p=self.delta_p_global,
        )

    def export_flow_field(self, output_dir: str | Path = "generated_flow_cases_3d", prefix: str = "flow3d") -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{prefix}_3d_full_flow_field.npz"
        utheta_abs_hat = self.UT if self.absolute_frame else self.UT + self._solid_ut()
        np.savez_compressed(
            file_path,
            R=self.R.detach().cpu().numpy(),
            Theta=self.Theta.detach().cpu().numpy(),
            Z=self.Z.detach().cpu().numpy(),
            phi=self.phi.detach().cpu().numpy(),
            UR=self.UR.detach().cpu().numpy(),
            UT=self.UT.detach().cpu().numpy(),
            UTheta=self.UT.detach().cpu().numpy(),
            UTheta_absolute=utheta_abs_hat.detach().cpu().numpy(),
            UZ=self.UZ.detach().cpu().numpy(),
            P=self.P.detach().cpu().numpy(),
            u_r=(self.UR * self.u_omega).detach().cpu().numpy(),
            u_theta=(self.UT * self.u_omega).detach().cpu().numpy(),
            u_theta_absolute=(utheta_abs_hat * self.u_omega).detach().cpu().numpy(),
            u_z=(self.UZ * self.u_zo).detach().cpu().numpy(),
            p=(self.P * self.P0).detach().cpu().numpy(),
            absolute_frame=np.asarray(self.absolute_frame, dtype=bool),
            initialization=np.asarray(self.initialization),
            flow_boundary_mode=np.asarray(self.flow_boundary_mode),
            balance_inlet_flow=np.asarray(self.balance_inlet_flow, dtype=bool),
            inlet_q_hat=np.asarray(self.inlet_flow_rate_hat(), dtype=float),
            outlet_q_hat=np.asarray(self.outlet_flow_rate_hat(), dtype=float),
            ibm_C=np.asarray(self.ibm_C, dtype=float),
            ibm_epsilon=np.asarray(self.ibm_epsilon, dtype=float),
            history_iteration=np.asarray([x["iteration"] for x in self.iteration_history], dtype=float),
            history_momentum=np.asarray([x["momentum"] for x in self.iteration_history], dtype=float),
            history_continuity=np.asarray([x["continuity"] for x in self.iteration_history], dtype=float),
            history_update=np.asarray([x["update"] for x in self.iteration_history], dtype=float),
            history_q_hat=np.asarray([x["q_hat"] for x in self.iteration_history], dtype=float),
            history_delta_p=np.asarray([x["delta_p"] for x in self.iteration_history], dtype=float),
        )
        return file_path

    def plot_convergence(self, output_dir: str | Path = "generated_flow_cases_3d", prefix: str = "flow3d") -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{prefix}_convergence.png"
        if not self.iteration_history:
            raise RuntimeError("No iteration history to plot.")
        it = np.asarray([x["iteration"] for x in self.iteration_history], dtype=float)
        mom = np.clip(np.asarray([x["momentum"] for x in self.iteration_history], dtype=float), 1e-30, None)
        mass = np.clip(np.asarray([x["continuity"] for x in self.iteration_history], dtype=float), 1e-30, None)
        plt.figure(figsize=(7, 4.5))
        plt.semilogy(it, mom, label="momentum")
        plt.semilogy(it, mass, label="continuity")
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path, dpi=170)
        plt.close()
        return file_path


def _npz_pick(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    """从 npz 中按候选字段名取第一个存在的数组。"""
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    raise KeyError(f"None of the fields exists in npz: {', '.join(names)}")


def _npz_optional(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    """读取可选字段，避免可视化旧格式 npz 时因为缺字段直接中断。"""
    if name not in data.files:
        return None
    return np.asarray(data[name])


def _npz_span_to_index(span: float, n: int) -> int:
    """把 0-1 span 位置映射到径向网格下标。"""
    return int(round(float(np.clip(span, 0.0, 1.0)) * (n - 1)))


def _npz_field_stats(field: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """返回 mask 内 mean/min/max；没有有效点时返回 NaN。"""
    values = np.asarray(field)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def visualize_npz_flow_field(
    npz_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    prefix: str | None = None,
    spans: tuple[float, ...] = (0.2, 0.5, 0.8),
    phi_cutoff: float = 0.05,
    show: bool = False,
) -> dict[str, Path]:
    """可视化 DataGenerator3D 导出的 npz 流场文件。

    Parameters
    ----------
    npz_path:
        `BladeCalc3D.export_flow_field()` 生成的 `.npz` 路径。函数也兼容包含
        `UR/UT/UZ/P` 或 `UR/UTheta/UZ/P` 的旧字段名。
    output_dir:
        图片输出目录；默认写到 npz 文件所在目录。
    prefix:
        图片文件名前缀；默认使用 npz 文件名。
    spans:
        需要展示的径向 span 位置，范围为 0-1。
    phi_cutoff:
        `phi <= phi_cutoff` 的区域会作为固体区遮罩。
    show:
        是否弹出 Matplotlib 窗口。批处理生成数据时建议保持 False。

    Returns
    -------
    dict[str, Path]
        已保存图片路径。至少包含 `span_plot`；如果 npz 中包含历史数据，还会包含
        `convergence_plot`。
    """
    npz_path = Path(npz_path)
    output_path = Path(output_dir) if output_dir is not None else npz_path.parent
    output_path.mkdir(parents=True, exist_ok=True)
    stem = prefix or npz_path.stem

    with np.load(npz_path) as data:
        # 优先画物理量。若文件来自手工构造或旧版本，只包含无量纲字段，也能退回显示。
        has_physical = all(name in data.files for name in ("u_r", "u_theta", "u_z", "p"))
        if has_physical:
            fields = {
                "UR": np.asarray(data["u_r"]),
                "UT": np.asarray(data["u_theta_absolute"] if "u_theta_absolute" in data.files else data["u_theta"]),
                "UZ": np.asarray(data["u_z"]),
                "P": np.asarray(data["p"]),
            }
            units = {"UR": "m/s", "UT": "m/s", "UZ": "m/s", "P": "Pa"}
            title_suffix = "physical"
        else:
            fields = {
                "UR": _npz_pick(data, "UR", "Ur", "ur"),
                "UT": _npz_pick(data, "UT", "UTheta", "Ut", "u_theta"),
                "UZ": _npz_pick(data, "UZ", "Uz", "uz"),
                "P": _npz_pick(data, "P", "p"),
            }
            units = {"UR": "dimensionless", "UT": "dimensionless", "UZ": "dimensionless", "P": "dimensionless"}
            title_suffix = "dimensionless"

        shape = fields["P"].shape
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D flow field, got P.shape={shape}.")
        nr, _, _ = shape
        phi = _npz_optional(data, "phi")
        if phi is not None and phi.shape == shape:
            solid_mask = np.asarray(phi <= phi_cutoff, dtype=bool)
        else:
            solid_mask = np.zeros(shape, dtype=bool)
        fluid_mask = ~solid_mask

        print("\n========== NPZ Flow Field Visualization ==========")
        print(f"file={npz_path}")
        print(f"shape={shape}, fields={title_suffix}")
        if "history_q_hat" in data.files and len(data["history_q_hat"]) > 0:
            print(f"last q_hat={float(data['history_q_hat'][-1]):.6g}")
        for span in spans:
            i = _npz_span_to_index(span, nr)
            fluid = fluid_mask[i]
            print(f"span={span:.2f} (i={i})")
            for name in ("UR", "UT", "UZ", "P"):
                mean_v, min_v, max_v = _npz_field_stats(fields[name][i], fluid)
                print(f"  {name} [{units[name]}] mean/min/max = {mean_v:.6g} / {min_v:.6g} / {max_v:.6g}")
        print("==================================================\n")

        # Span 切片图：沿径向取若干 Theta-Z 面，和 SurrogateModeling.py 的 post 风格一致。
        span_path = output_path / f"{stem}_npz_spans.png"
        fig, axes = plt.subplots(len(spans), 4, figsize=(18, 3.6 * len(spans)), squeeze=False)
        cmaps = {"UR": "coolwarm", "UT": "coolwarm", "UZ": "viridis", "P": "plasma"}
        for row, span in enumerate(spans):
            i = _npz_span_to_index(span, nr)
            blade_mask = solid_mask[i].T
            for col, name in enumerate(("UR", "UT", "UZ", "P")):
                ax = axes[row, col]
                image_data = np.ma.array(fields[name][i].T, mask=blade_mask)
                image = ax.imshow(image_data, origin="lower", aspect="auto", cmap=cmaps[name])
                if np.any(blade_mask):
                    ax.contour(blade_mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
                ax.set_title(f"{name} @ span={span:.2f} ({title_suffix})")
                ax.set_xlabel("Theta index")
                ax.set_ylabel("Z index")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=units[name])
        fig.tight_layout()
        fig.savefig(span_path, dpi=170, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

        paths: dict[str, Path] = {"span_plot": span_path}

        # 如果 npz 带收敛历史，则额外画 residual / q_hat / delta_p。
        history_iteration = _npz_optional(data, "history_iteration")
        history_momentum = _npz_optional(data, "history_momentum")
        history_continuity = _npz_optional(data, "history_continuity")
        if history_iteration is not None and history_momentum is not None and history_continuity is not None:
            if len(history_iteration) > 0:
                conv_path = output_path / f"{stem}_npz_convergence.png"
                fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
                axes[0].semilogy(history_iteration, np.clip(history_momentum, 1e-30, None), label="momentum")
                axes[0].semilogy(history_iteration, np.clip(history_continuity, 1e-30, None), label="continuity")
                axes[0].set_ylabel("Residual")
                axes[0].grid(True, which="both", alpha=0.25)
                axes[0].legend()

                q_hat = _npz_optional(data, "history_q_hat")
                delta_p = _npz_optional(data, "history_delta_p")
                if q_hat is not None and len(q_hat) == len(history_iteration):
                    axes[1].plot(history_iteration, q_hat, label="q_hat")
                    axes[1].axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.55)
                if delta_p is not None and len(delta_p) == len(history_iteration):
                    axes[1].plot(history_iteration, delta_p, label="delta_p")
                axes[1].set_xlabel("Iteration")
                axes[1].set_ylabel("Flow control")
                axes[1].grid(True, alpha=0.25)
                axes[1].legend()
                fig.tight_layout()
                fig.savefig(conv_path, dpi=170, bbox_inches="tight")
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                paths["convergence_plot"] = conv_path

    return paths


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_couple_3d_main() -> SolveLog3D:
    seed_everything(10492)
    solver = BladeCalc3D(
        n=48,
        max_iter=240,
        pressure_solver="gmg",
        pressure_max_inner=500,
        pressure_tol=1e-7,
        device="cuda",
        delta_p_initial=0.1,
        initialization="fluent_hybrid",
        initialization_sweeps=8,
    )
    print(f"3D initialization ({solver.initialization}): {solver.initialization_summary()}")
    log = solver.solve(method="couple", report_interval=1)
    out_dir = Path("generated_flow_cases_3d")
    field_path = solver.export_flow_field(out_dir, "couple3d")
    conv_path = solver.plot_convergence(out_dir, "couple3d")
    print(f"3D field saved to: {field_path}")
    print(f"3D convergence plot saved to: {conv_path}")
    return log


def run_simple_3d_main() -> SolveLog3D:
    seed_everything(10492)
    solver = BladeCalc3D(
        n=48,
        max_iter=240,
        pressure_solver="gmg",
        pressure_max_inner=300,
        pressure_tol=1e-7,
        device="cuda",
        delta_p_initial=0.1,
        initialization="fluent_hybrid",
        initialization_sweeps=8,
    )
    print(f"3D initialization ({solver.initialization}): {solver.initialization_summary()}")
    log = solver.solve(method="simple", max_outer=8, inner_per_outer=30, report_interval=1)
    out_dir = Path("generated_flow_cases_3d")
    field_path = solver.export_flow_field(out_dir, "simple3d")
    conv_path = solver.plot_convergence(out_dir, "simple3d")
    print(f"3D field saved to: {field_path}")
    print(f"3D convergence plot saved to: {conv_path}")
    return log


def run_main(method: str = "couple") -> SolveLog3D:
    method = method.lower().replace("-", "_")
    if method in {"simple", "simple3d", "simple_3d"}:
        return run_simple_3d_main()
    if method in {"couple", "couple3d", "couple_3d", "gmg"}:
        return run_couple_3d_main()
    raise ValueError("method must be 'couple' or 'simple'.")


if __name__ == "__main__":
    print(run_main("simple"))
    visualize_npz_flow_field("./generated_flow_cases_3d/simple3d_3d_full_flow_field.npz")
