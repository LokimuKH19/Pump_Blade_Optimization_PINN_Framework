from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from BladeImport import attach_blade_to_solver
import matplotlib.pyplot as plt
import numpy as np
from PressureUpdaters import PressureUpdater
import torch


@dataclass
class SolveLog:
    method: str
    converged: bool
    iterations: int
    final_momentum: float
    final_mass: float
    final_update: float
    final_q_hat: float
    delta_p: float


class BladeCalc:
    """Slice-wise 2D cylindrical-layer flow generator.

    The R direction is treated as a batch of independent cylindrical layers.
    Each layer solves a collocated FVM system on the Theta-Z plane, using
    Rhie-Chow interpolation and SIMPLE pressure correction.
    """

    def __init__(
        self,
        n: int = 64,
        rh: float = 2.0,
        rs: float = 4.0,
        h: float = 8.0,
        mu: float = 1.0,
        rho: float = 1.0,
        omega: float = 1.0,
        qv: float = 1.0,
        n_blade: int = 1,
        max_iter: int = 5000,
        tol: float = 1e-6,
        u_relax: float = 0.5,
        p_relax: float = 0.3,
        device: str = "cuda",
        z0: float = 0.0,
        blade_params: str | Path | None = "blade_params.json",
        absolute_frame: bool = True,
        delta_p_initial: float = 0.0,
        pressure_solver: str = "bicgstab",
        pressure_max_inner: int = 600,
        pressure_tol: float = 1e-8,
        ibm_C: float = 1.0,
        ibm_epsilon: float = 0.025,
        pseudo_dt: float = 0.001,
        rans_model: str = "none",
        rans_mixing_length: float = 0.08,
        rans_nut_max_ratio: float = 250.0,
        local_pseudo_dt: bool = False,
        pseudo_cfl: float = 2.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.g = 9.8
        self.n = int(n)
        if self.n < 48:
            print(
                f"Warning: n={self.n} is intended only for debugging. "
                "IBM blade geometry is under-resolved on coarse grids, so convergence behavior is not representative."
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

        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.u_relax = float(u_relax)
        self.p_relax = float(p_relax)
        self.pressure_solver = str(pressure_solver).lower()
        self.pressure_max_inner = int(pressure_max_inner)
        self.pressure_tol = float(pressure_tol)
        self.pseudo_dt = max(float(pseudo_dt), 1e-6)
        self.local_pseudo_dt = bool(local_pseudo_dt)
        self.pseudo_cfl = max(float(pseudo_cfl), 1e-6)

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
        self.delta_p_global = float(delta_p_initial)
        self.delta_p_min = 1e-8
        self.delta_p_max = 1e8
        self.outer_flow_tol = 2e-3
        self.delta_p_relax = 0.55
        self.couple_relax = 0.35
        self.gmg_levels = 5
        self.gmg_pre_smooth = 3
        self.gmg_post_smooth = 3
        self.gmg_cycles = 8
        self.gmg_omega = 0.72
        self.couple_pressure_sweeps = 3
        self.pressure_projection_relax = 0.8
        self.couple_momentum_update_limit = 0.35
        self.couple_pressure_velocity_limit = 0.25
        self.couple_pressure_update_limit = 0.75
        self.couple_field_abs_limit = 50.0
        self.couple_flow_control = False
        self.couple_flow_control_interval = 20
        self.couple_flow_control_gain = 0.18
        self.couple_flow_control_clip = 0.12
        self.momentum_sweeps = 3
        self.momentum_solver_relax = 0.75
        self.couple_pressure_backtracking = True
        self.couple_pressure_max_momentum_growth = 1.25
        self.couple_pressure_min_relax = 1e-3
        self.couple_pressure_interval = 1
        self.rans_model = str(rans_model).lower()
        self.rans_mixing_length = max(float(rans_mixing_length), 0.0)
        self.rans_nut_max_ratio = max(float(rans_nut_max_ratio), 0.0)
        self.rans_smoothing_steps = 1
        self.ibm_hard_phi = 0.05
        self.ibm_C = float(ibm_C)
        self.ibm_epsilon = float(ibm_epsilon)

        self._build_grid()
        self._build_geometry()
        self._init_fields()
        self._init_blade_boundary()
        self._apply_boundary()

        self.A_theta = torch.ones_like(self.P)
        self.A_z = torch.ones_like(self.P)
        self.P_prime = torch.zeros_like(self.P)
        self.pressure_updater = PressureUpdater(self)
        self.nut_ratio = torch.zeros_like(self.P)
        self.UT_tilde = self.UT.clone()
        self.Uz_tilde = self.Uz.clone()
        self.last_history: list[dict[str, float]] = []
        self.iteration_history: list[dict[str, float]] = []
        self.last_solve_method = ""
        self._couple_step_count = 0

    def _build_grid(self) -> None:
        n = self.n
        if n < 4:
            raise ValueError("n must be at least 4 for the Theta-Z stencil.")
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
        self.dA = torch.full_like(self.r_hatC, self.dTheta * self.dZ)
        self.theta_area = torch.full_like(self.r_hatC, self.dZ)
        self.z_area = torch.full_like(self.r_hatC, self.dTheta)

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
        self.UT = torch.zeros(shape, device=self.device)
        self.Uz = torch.ones(shape, device=self.device)
        self.blade_mask = None
        self.blade_distance = None
        self.blade_distance_z = None
        self.blade_footprint = None
        self.blade_boundary_meta = None
        self.blade_boundary_band_cells = 1.5
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
        signed_distance = torch.as_tensor(self.boundary.signed_distance, dtype=torch.float32, device=self.device)
        eps2 = max(self.ibm_epsilon**2, 1e-20)
        self.phi = 1.0 - torch.exp(-self.ibm_C * signed_distance**2 / eps2)
        self.phi = torch.clamp(self.phi, 0.0, 1.0)
        self.phi_mask = self.phi

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

    def attach_blade_boundary(self, blade_params_path: str | Path = "blade_params.json", band_cells: float = 1.5):
        return attach_blade_to_solver(self, blade_params_path, band_cells=band_cells)

    def _solid_ut(self) -> torch.Tensor:
        if not self.absolute_frame:
            return torch.zeros_like(self.r_hatC)
        return self.sgn_omega * self.delta * self.r_hatC

    def _hard_solid_mask(self) -> torch.Tensor:
        return self.phi <= self.ibm_hard_phi

    def _theta_plus(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        out[:, :-1, :] = x[:, 1:, :]
        out[:, -1, :] = x[:, 1, :]
        return out

    def _theta_minus(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        out[:, 1:, :] = x[:, :-1, :]
        out[:, 0, :] = x[:, -2, :]
        return out

    def _z_plus(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        out[:, :, :-1] = x[:, :, 1:]
        out[:, :, -1] = x[:, :, -1]
        return out

    def _z_minus(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        out[:, :, 1:] = x[:, :, :-1]
        out[:, :, 0] = x[:, :, 0]
        return out

    def neighbor(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        if direction == "N":
            return self._theta_plus(x)
        if direction == "S":
            return self._theta_minus(x)
        if direction == "T":
            return self._z_plus(x)
        if direction == "B":
            return self._z_minus(x)
        raise ValueError(f"unknown direction {direction!r}")

    def _project_theta_periodic(self, *fields: torch.Tensor) -> None:
        for field in fields:
            field[:, -1, :] = field[:, 0, :]

    def _set_pressure_boundary(self) -> None:
        self.P[:, :, 0] = self.delta_p_global
        self.P[:, :, -1] = 0.0
        self._project_theta_periodic(self.P)

    def _set_velocity_boundary(self) -> None:
        if self.n > 2:
            self.UT[:, :, 0] = self.UT[:, :, 1]
            self.UT[:, :, -1] = self.UT[:, :, -2]
            self.Uz[:, :, 0] = self.Uz[:, :, 1]
            self.Uz[:, :, -1] = self.Uz[:, :, -2]
        self._project_theta_periodic(self.UT, self.Uz)

    def _apply_ibm(self) -> None:
        solid_ut = self._solid_ut()
        hard_solid = self._hard_solid_mask()
        self.UT = torch.where(hard_solid, solid_ut, self.UT)
        self.Uz = torch.where(hard_solid, torch.zeros_like(self.Uz), self.Uz)
        if hasattr(self, "UT_tilde"):
            self.UT_tilde = torch.where(hard_solid, solid_ut, self.UT_tilde)
            self.Uz_tilde = torch.where(hard_solid, torch.zeros_like(self.Uz_tilde), self.Uz_tilde)

    def _apply_boundary(self) -> None:
        self._set_pressure_boundary()
        self._set_velocity_boundary()
        self._apply_ibm()
        self._project_theta_periodic(self.UT, self.Uz, self.P)

    def _pressure_gradient_theta(self, P: torch.Tensor) -> torch.Tensor:
        return (self.neighbor(P, "N") - self.neighbor(P, "S")) / (2.0 * self.dTheta)

    def _pressure_gradient_z(self, P: torch.Tensor) -> torch.Tensor:
        grad = (self.neighbor(P, "T") - self.neighbor(P, "B")) / (2.0 * self.dZ)
        grad[:, :, 0] = (P[:, :, 1] - P[:, :, 0]) / self.dZ
        grad[:, :, -1] = (P[:, :, -1] - P[:, :, -2]) / self.dZ
        return grad

    def _effective_inv_reynolds(self) -> torch.Tensor:
        inv_re = 1.0 / max(self.Re_omega, 1e-12)
        if self.rans_model in {"none", "off", "laminar"} or self.rans_mixing_length <= 0.0:
            self.nut_ratio = torch.zeros_like(self.P)
            return torch.full_like(self.P, inv_re)

        dUT_dtheta = self._pressure_gradient_theta(self.UT)
        dUT_dz = self._pressure_gradient_z(self.UT)
        dUz_dtheta = self._pressure_gradient_theta(self.Uz)
        dUz_dz = self._pressure_gradient_z(self.Uz)

        s_tt = self.K_theta_C * dUT_dtheta
        s_zz = self.Ku * self.Lambda * dUz_dz
        s_tz = self.Lambda * dUT_dz + self.Ku * self.K_theta_C * dUz_dtheta
        strain = torch.sqrt(torch.clamp(2.0 * (s_tt**2 + s_zz**2) + s_tz**2, min=0.0) + 1e-20)

        mixing_length = torch.full_like(strain, self.rans_mixing_length)
        if self.blade_distance is not None:
            wall_distance = torch.clamp(self.blade_distance, min=0.0)
            mixing_length = torch.minimum(mixing_length, 0.41 * wall_distance + min(self.dTheta, self.dZ))

        damping = torch.clamp(self.phi, 0.0, 1.0) ** 2
        nut_ratio = (mixing_length**2) * strain * self.Re_omega * damping
        nut_ratio = torch.clamp(nut_ratio, min=0.0, max=self.rans_nut_max_ratio)
        for _ in range(max(int(self.rans_smoothing_steps), 0)):
            nut_ratio = (
                0.5 * nut_ratio
                + 0.125
                * (
                    self.neighbor(nut_ratio, "N")
                    + self.neighbor(nut_ratio, "S")
                    + self.neighbor(nut_ratio, "T")
                    + self.neighbor(nut_ratio, "B")
                )
            )
            nut_ratio = torch.clamp(nut_ratio, min=0.0, max=self.rans_nut_max_ratio)
            self._project_theta_periodic(nut_ratio)

        self.nut_ratio = nut_ratio
        return inv_re * (1.0 + nut_ratio)

    def rhie_chow(
        self,
        UT: torch.Tensor,
        Uz: torch.Tensor,
        P: torch.Tensor,
        A_theta: torch.Tensor,
        A_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A_theta = A_theta + (1.0 - self.phi) * 1.0 + 1e-6
        A_z = A_z + (1.0 - self.phi) * 1.0 + 1e-6
        P_n = self.neighbor(P, "N")
        P_s = self.neighbor(P, "S")
        P_t = self.neighbor(P, "T")
        P_b = self.neighbor(P, "B")

        UT_n_cell = self.neighbor(UT, "N")
        UT_s_cell = self.neighbor(UT, "S")
        Uz_t_cell = self.neighbor(Uz, "T")
        Uz_b_cell = self.neighbor(Uz, "B")

        A_theta_n = 0.5 * (A_theta + self.neighbor(A_theta, "N"))
        A_theta_s = 0.5 * (A_theta + self.neighbor(A_theta, "S"))
        A_z_t = 0.5 * (A_z + self.neighbor(A_z, "T"))
        A_z_b = 0.5 * (A_z + self.neighbor(A_z, "B"))

        grad_theta_c = self._pressure_gradient_theta(P)
        grad_z_c = self._pressure_gradient_z(P)
        grad_theta_n = self.neighbor(grad_theta_c, "N")
        grad_theta_s = self.neighbor(grad_theta_c, "S")
        grad_z_t = self.neighbor(grad_z_c, "T")
        grad_z_b = self.neighbor(grad_z_c, "B")

        gt_c = self.dA * self.Eu_omega * self.K_theta_C * grad_theta_c
        gz_c = self.dA * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * grad_z_c
        gt_n_cell = self.neighbor(gt_c, "N")
        gt_s_cell = self.neighbor(gt_c, "S")
        gz_t_cell = self.neighbor(gz_c, "T")
        gz_b_cell = self.neighbor(gz_c, "B")

        gt_n_face = self.dA * self.Eu_omega * self.K_theta_C * (P_n - P) / self.dTheta
        gt_s_face = self.dA * self.Eu_omega * self.K_theta_C * (P - P_s) / self.dTheta
        gz_t_face = self.dA * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * (P_t - P) / self.dZ
        gz_b_face = self.dA * self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * (P - P_b) / self.dZ

        inv_theta = 1.0 / torch.clamp(A_theta, min=1e-12)
        inv_z = 1.0 / torch.clamp(A_z, min=1e-12)

        UT_n = 0.5 * (UT + UT_n_cell) - (
            gt_n_face / torch.clamp(A_theta_n, min=1e-12)
            - 0.5 * (gt_c * inv_theta + gt_n_cell / torch.clamp(self.neighbor(A_theta, "N"), min=1e-12))
        )
        UT_s = 0.5 * (UT + UT_s_cell) - (
            gt_s_face / torch.clamp(A_theta_s, min=1e-12)
            - 0.5 * (gt_c * inv_theta + gt_s_cell / torch.clamp(self.neighbor(A_theta, "S"), min=1e-12))
        )
        Uz_t = 0.5 * (Uz + Uz_t_cell) - (
            gz_t_face / torch.clamp(A_z_t, min=1e-12)
            - 0.5 * (gz_c * inv_z + gz_t_cell / torch.clamp(self.neighbor(A_z, "T"), min=1e-12))
        )
        Uz_b = 0.5 * (Uz + Uz_b_cell) - (
            gz_b_face / torch.clamp(A_z_b, min=1e-12)
            - 0.5 * (gz_c * inv_z + gz_b_cell / torch.clamp(self.neighbor(A_z, "B"), min=1e-12))
        )

        Uz_t[:, :, -1] = Uz[:, :, -1]
        Uz_b[:, :, 0] = Uz[:, :, 0]
        self._project_theta_periodic(UT_n, UT_s, Uz_t, Uz_b)
        return UT_n, UT_s, Uz_t, Uz_b

    def _face_fluxes(
        self,
        UT_face_n: torch.Tensor,
        UT_face_s: torch.Tensor,
        Uz_face_t: torch.Tensor,
        Uz_face_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Fn = self.K_theta_C * self.theta_area * UT_face_n
        Fs = -self.K_theta_C * self.theta_area * UT_face_s
        Ft = self.Lambda * self.Ku * self.z_area * Uz_face_t
        Fb = -self.Lambda * self.Ku * self.z_area * Uz_face_b
        return Fn, Fs, Ft, Fb

    def _face_opening(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phi_n = torch.minimum(self.phi, self.neighbor(self.phi, "N"))
        phi_s = torch.minimum(self.phi, self.neighbor(self.phi, "S"))
        phi_t = torch.minimum(self.phi, self.neighbor(self.phi, "T"))
        phi_b = torch.minimum(self.phi, self.neighbor(self.phi, "B"))
        phi_t[:, :, -1] = self.phi[:, :, -1]
        phi_b[:, :, 0] = self.phi[:, :, 0]
        self._project_theta_periodic(phi_n, phi_s, phi_t, phi_b)
        return phi_n, phi_s, phi_t, phi_b

    def _assemble_momentum_coefficients(self) -> dict[str, torch.Tensor]:
        UT_n_face, UT_s_face, Uz_t_face, Uz_b_face = self.rhie_chow(
            self.UT,
            self.Uz,
            self.P,
            self.A_theta,
            self.A_z,
        )
        Fn, Fs, Ft, Fb = self._face_fluxes(UT_n_face, UT_s_face, Uz_t_face, Uz_b_face)
        phi_n, phi_s, phi_t, phi_b = self._face_opening()
        Fn = Fn * phi_n
        Fs = Fs * phi_s
        Ft = Ft * phi_t
        Fb = Fb * phi_b

        inv_re = self._effective_inv_reynolds()
        inv_re_n = 0.5 * (inv_re + self.neighbor(inv_re, "N"))
        inv_re_s = 0.5 * (inv_re + self.neighbor(inv_re, "S"))
        inv_re_t = 0.5 * (inv_re + self.neighbor(inv_re, "T"))
        inv_re_b = 0.5 * (inv_re + self.neighbor(inv_re, "B"))
        Dn = phi_n * (self.K_theta_C**2) * self.theta_area * inv_re_n / self.dTheta
        Ds = phi_s * (self.K_theta_C**2) * self.theta_area * inv_re_s / self.dTheta
        Dt = phi_t * (self.Lambda**2) * self.z_area * inv_re_t / self.dZ
        Db = phi_b * (self.Lambda**2) * self.z_area * inv_re_b / self.dZ

        aN = Dn - torch.clamp(Fn, max=0.0)
        aS = Ds - torch.clamp(Fs, max=0.0)
        aT = Dt - torch.clamp(Ft, max=0.0)
        aB = Db - torch.clamp(Fb, max=0.0)
        aP_conv = (
            torch.clamp(Fn, min=0.0)
            + torch.clamp(Fs, min=0.0)
            + torch.clamp(Ft, min=0.0)
            + torch.clamp(Fb, min=0.0)
        )
        aP_diff = Dn + Ds + Dt + Db

        solid = 1.0 - self.phi
        pseudo = self.dA / self.pseudo_dt
        if self.local_pseudo_dt:
            spectral_radius = (
                torch.abs(Fn)
                + torch.abs(Fs)
                + torch.abs(Ft)
                + torch.abs(Fb)
                + 2.0 * aP_diff
                + solid
            )
            pseudo = pseudo + spectral_radius / self.pseudo_cfl
        A_theta = aP_conv + aP_diff + self.dA * inv_re / torch.clamp(self.r_hatC**2, min=1e-12) + 1e-12
        A_z = aP_conv + aP_diff + 1e-12
        A_theta = A_theta + solid + pseudo
        A_z = A_z + solid + pseudo

        return {
            "N": aN,
            "S": aS,
            "T": aT,
            "B": aB,
            "A_theta": A_theta,
            "A_z": A_z,
            "pseudo": pseudo,
        }

    def momentum(self) -> None:
        coef = self._assemble_momentum_coefficients()
        self.A_theta = coef["A_theta"]
        self.A_z = coef["A_z"]

        pressure_theta = self.Eu_omega * self.K_theta_C * self._pressure_gradient_theta(self.P)
        pressure_z = self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * self._pressure_gradient_z(self.P)

        UT_ref = self.UT.clone()
        Uz_ref = self.Uz.clone()
        UT_iter = self.UT.clone()
        Uz_iter = self.Uz.clone()
        solid = 1.0 - self.phi
        solid_ut = self._solid_ut()
        hard_solid = self._hard_solid_mask()
        relax = float(np.clip(self.momentum_solver_relax, 0.05, 1.0))

        for _ in range(max(int(self.momentum_sweeps), 1)):
            UT_n = self.neighbor(UT_iter, "N")
            UT_s = self.neighbor(UT_iter, "S")
            UT_t = self.neighbor(UT_iter, "T")
            UT_b = self.neighbor(UT_iter, "B")
            Uz_n = self.neighbor(Uz_iter, "N")
            Uz_s = self.neighbor(Uz_iter, "S")
            Uz_t = self.neighbor(Uz_iter, "T")
            Uz_b = self.neighbor(Uz_iter, "B")

            bf_theta = coef["N"] * UT_n + coef["S"] * UT_s + coef["T"] * UT_t + coef["B"] * UT_b
            bf_z = coef["N"] * Uz_n + coef["S"] * Uz_s + coef["T"] * Uz_t + coef["B"] * Uz_b
            bf_theta = bf_theta + solid * solid_ut + coef["pseudo"] * UT_ref
            bf_z = bf_z + coef["pseudo"] * Uz_ref

            UT_new = (bf_theta - self.dA * pressure_theta) / torch.clamp(self.A_theta, min=1e-12)
            Uz_new = (bf_z - self.dA * pressure_z - self.dA * self.G_star) / torch.clamp(self.A_z, min=1e-12)
            UT_new = torch.where(hard_solid, solid_ut, UT_new)
            Uz_new = torch.where(hard_solid, torch.zeros_like(Uz_new), Uz_new)
            UT_iter = UT_iter + relax * (UT_new - UT_iter)
            Uz_iter = Uz_iter + relax * (Uz_new - Uz_iter)
            self._project_theta_periodic(UT_iter, Uz_iter)

        self.UT_tilde = UT_iter
        self.Uz_tilde = Uz_iter
        self._project_theta_periodic(self.UT_tilde, self.Uz_tilde)

    def pressure(self) -> None:
        self.pressure_updater.simple_step()

    def _linear_flux_divergence(self, UT: torch.Tensor, Uz: torch.Tensor) -> torch.Tensor:
        UT_n = 0.5 * (UT + self.neighbor(UT, "N"))
        UT_s = 0.5 * (UT + self.neighbor(UT, "S"))
        Uz_t = 0.5 * (Uz + self.neighbor(Uz, "T"))
        Uz_b = 0.5 * (Uz + self.neighbor(Uz, "B"))
        Uz_t[:, :, -1] = Uz[:, :, -1]
        Uz_b[:, :, 0] = Uz[:, :, 0]
        Fn, Fs, Ft, Fb = self._face_fluxes(UT_n, UT_s, Uz_t, Uz_b)
        phi_n, phi_s, phi_t, phi_b = self._face_opening()
        return Fn * phi_n + Fs * phi_s + Ft * phi_t + Fb * phi_b

    def _rhie_flux_divergence(self, UT: torch.Tensor, Uz: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        UT_n_face, UT_s_face, Uz_t_face, Uz_b_face = self.rhie_chow(
            UT,
            Uz,
            P,
            self.A_theta,
            self.A_z,
        )
        Fn, Fs, Ft, Fb = self._face_fluxes(UT_n_face, UT_s_face, Uz_t_face, Uz_b_face)
        phi_n, phi_s, phi_t, phi_b = self._face_opening()
        return Fn * phi_n + Fs * phi_s + Ft * phi_t + Fb * phi_b

    def _momentum_residual(self, coef: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        UT_n = self.neighbor(self.UT, "N")
        UT_s = self.neighbor(self.UT, "S")
        UT_t = self.neighbor(self.UT, "T")
        UT_b = self.neighbor(self.UT, "B")
        Uz_n = self.neighbor(self.Uz, "N")
        Uz_s = self.neighbor(self.Uz, "S")
        Uz_t = self.neighbor(self.Uz, "T")
        Uz_b = self.neighbor(self.Uz, "B")

        pressure_theta = self.Eu_omega * self.K_theta_C * self._pressure_gradient_theta(self.P)
        pressure_z = self.Eu_omega * (self.Lambda / max(self.Ku, 1e-12)) * self._pressure_gradient_z(self.P)
        solid = 1.0 - self.phi

        res_theta = (
            coef["A_theta"] * self.UT
            - coef["N"] * UT_n
            - coef["S"] * UT_s
            - coef["T"] * UT_t
            - coef["B"] * UT_b
            - coef["pseudo"] * self.UT
            - solid * self._solid_ut()
            + self.dA * pressure_theta
        )    # 角向残差
        res_z = (
            coef["A_z"] * self.Uz
            - coef["N"] * Uz_n
            - coef["S"] * Uz_s
            - coef["T"] * Uz_t
            - coef["B"] * Uz_b
            - coef["pseudo"] * self.Uz
            + self.dA * pressure_z
            + self.dA * self.G_star
        )    # 轴向残差
        return res_theta, res_z

    def _fluid_residual_mask(self) -> torch.Tensor:
        mask = self.phi > 0.2
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

    def _velocity_update_scale(self, dUT: torch.Tensor, dUz: torch.Tensor, limit: float | None) -> float:
        if limit is None or limit <= 0.0:
            return 1.0
        mask = self.phi > 0.2
        if torch.any(mask):
            max_update = torch.maximum(
                torch.max(torch.abs(dUT[mask])),
                torch.max(torch.abs(dUz[mask])),
            )
        else:
            max_update = torch.maximum(torch.max(torch.abs(dUT)), torch.max(torch.abs(dUz)))
        max_value = float(max_update.item())
        if not np.isfinite(max_value) or max_value <= limit:
            return 1.0 if np.isfinite(max_value) else 0.0
        return max(float(limit) / max(max_value, 1e-30), 0.0)

    def _pressure_update_scale(self, dP: torch.Tensor, limit: float | None) -> float:
        if limit is None or limit <= 0.0:
            return 1.0
        max_update = float(torch.max(torch.abs(dP)).item())
        if not np.isfinite(max_update) or max_update <= limit:
            return 1.0 if np.isfinite(max_update) else 0.0
        return max(float(limit) / max(max_update, 1e-30), 0.0)

    def _clamp_couple_fields(self) -> None:
        limit = float(self.couple_field_abs_limit)
        if limit <= 0.0:
            return
        self.UT = torch.clamp(self.UT, min=-limit, max=limit)
        self.Uz = torch.clamp(self.Uz, min=-limit, max=limit)
        p_limit = max(limit, 4.0 * abs(self.delta_p_global), 1.0)
        self.P = torch.clamp(self.P, min=-p_limit, max=p_limit)
        self._apply_boundary()

    def momentum_error(self) -> float:
        coef = self._assemble_momentum_coefficients()
        res_theta, res_z = self._momentum_residual(coef)
        mask = self._fluid_residual_mask()
        return max(self._masked_absmax(res_theta, mask), self._masked_absmax(res_z, mask))

    def couple_step(self) -> tuple[float, float]:
        step_count = int(getattr(self, "_couple_step_count", 0))
        UT_old = self.UT.clone()
        Uz_old = self.Uz.clone()
        P_old = self.P.clone()
        P_prime_old = self.P_prime.clone()

        self.momentum()
        inv_theta = self.phi / torch.clamp(self.A_theta + (1.0 - self.phi), min=1e-12)
        inv_z = self.phi / torch.clamp(self.A_z + (1.0 - self.phi), min=1e-12)

        dUT = self.couple_relax * (self.UT_tilde - UT_old)
        dUz = self.couple_relax * (self.Uz_tilde - Uz_old)
        scale = self._velocity_update_scale(dUT, dUz, self.couple_momentum_update_limit)
        self.UT = UT_old + scale * dUT
        self.Uz = Uz_old + scale * dUz
        self._apply_boundary()
        self._clamp_couple_fields()

        if not self._fields_are_finite(self.UT, self.Uz, self.P):
            self.UT = UT_old
            self.Uz = Uz_old
            self.P = P_old
            self.P_prime = P_prime_old
            self._apply_boundary()
            return 0.0, self.continuity_error()

        do_pressure = (
            self.couple_pressure_sweeps > 0
            and self.couple_pressure_interval > 0
            and step_count % self.couple_pressure_interval == 0
        )
        for _ in range(self.couple_pressure_sweeps if do_pressure else 0):
            beta = -self._rhie_flux_divergence(self.UT, self.Uz, self.P)
            base_mass = self.continuity_error()
            base_momentum = self.momentum_error()
            saved_state = (self.UT.clone(), self.Uz.clone(), self.P.clone(), self.P_prime.clone())
            relax = self.pressure_projection_relax
            accepted = False
            while relax >= self.couple_pressure_min_relax:
                self.pressure_updater.project(
                    inv_theta,
                    inv_z,
                    beta,
                    relax,
                    self.couple_pressure_velocity_limit,
                    self.couple_pressure_update_limit,
                )
                self._clamp_couple_fields()
                new_mass = self.continuity_error()
                new_momentum = self.momentum_error()
                finite = self._fields_are_finite(self.UT, self.Uz, self.P)
                momentum_limit = max(base_momentum * self.couple_pressure_max_momentum_growth, base_momentum + 1e-4)
                if finite and new_mass <= base_mass and new_momentum <= momentum_limit:
                    accepted = True
                    break
                self.UT, self.Uz, self.P, self.P_prime = (item.clone() for item in saved_state)
                self._apply_boundary()
                if not self.couple_pressure_backtracking:
                    break
                relax *= 0.5
            if not accepted:
                self.UT, self.Uz, self.P, self.P_prime = (item.clone() for item in saved_state)
                self._apply_boundary()
                break

        self._couple_step_count = step_count + 1
        update = torch.maximum(
            torch.max(torch.abs(self.UT - UT_old)),
            torch.max(torch.abs(self.Uz - Uz_old)),
        )
        mass = self.continuity_error()
        return float(update.item()), mass

    def simple_step(self) -> tuple[float, float, float]:
        UT_old = self.UT.clone()
        Uz_old = self.Uz.clone()
        self.momentum()
        self.pressure()
        self._apply_boundary()

        update = torch.maximum(
            torch.max(torch.abs(self.UT - UT_old)),
            torch.max(torch.abs(self.Uz - Uz_old)),
        )
        mass = self.continuity_error()
        momentum = self.momentum_error()
        return float(update.item()), mass, momentum

    # 计算连续性的情况（因为假设了没有r向流动因此其实没有完全封闭）
    def continuity_error(self) -> float:
        div = self._rhie_flux_divergence(self.UT, self.Uz, self.P)
        mask = self._fluid_residual_mask()
        if torch.any(mask):
            return float(torch.max(torch.abs(div[mask])).item())
        return float(torch.max(torch.abs(div)).item())

    def outlet_flow_rate_hat(self) -> float:
        fluid = self.phi[:, :, -1]
        weighted = self.Uz[:, :, -1] * fluid * self.outlet_area_weight
        q_raw = torch.sum(weighted)
        return float((q_raw / torch.clamp(self.outlet_area_target, min=1e-12)).item())

    def _nudge_delta_p(self, q_hat: float, prev: tuple[float, float] | None) -> tuple[float, float]:
        err = q_hat - 1.0
        old_dp = self.delta_p_global
        if prev is not None:
            prev_dp, prev_err = prev
            denom = err - prev_err
            if abs(denom) > 1e-8:
                candidate = old_dp - err * (old_dp - prev_dp) / denom
            else:
                candidate = old_dp * (1.0 - 0.5 * err)
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

    def solve(
        self,
        max_outer: int = 8,
        inner_per_outer: int | None = None,
        report_interval: int = 25,
        method: str = "simple",
    ) -> SolveLog:
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
        final_update = float("inf")
        final_mass = float("inf")
        final_momentum = float("inf")

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
                        "nut_max": float(torch.max(self.nut_ratio).item()) if hasattr(self, "nut_ratio") else 0.0,
                        "nut_mean": float(torch.mean(self.nut_ratio).item()) if hasattr(self, "nut_ratio") else 0.0,
                    }
                )
                if report_interval and (inner % report_interval == 0 or inner == inner_limit - 1):
                    print(
                        f"SIMPLE outer={outer:02d} inner={inner:04d} "
                        f"mom={final_momentum:.3e} mass={final_mass:.3e} "
                        f"update={final_update:.3e} "
                        f"q_hat={q_hat:.5f} dP={self.delta_p_global:.5e}"
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
        return SolveLog(
            method="SIMPLE",
            converged=converged,
            iterations=total_iter,
            final_momentum=final_momentum,
            final_mass=final_mass,
            final_update=final_update,
            final_q_hat=self.outlet_flow_rate_hat(),
            delta_p=self.delta_p_global,
        )

    def solve_couple(self, max_iter: int | None = None, report_interval: int = 25) -> SolveLog:
        limit = int(max_iter or self.max_iter)
        converged = False
        final_update = float("inf")
        final_mass = float("inf")
        final_momentum = float("inf")
        self.last_solve_method = "couple"
        self.iteration_history = []
        self._couple_step_count = 0
        for it in range(limit):
            final_update, final_mass = self.couple_step()
            final_momentum = self.momentum_error()
            q_hat = self.outlet_flow_rate_hat()
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
                    "nut_max": float(torch.max(self.nut_ratio).item()) if hasattr(self, "nut_ratio") else 0.0,
                    "nut_mean": float(torch.mean(self.nut_ratio).item()) if hasattr(self, "nut_ratio") else 0.0,
                }
            )
            if report_interval and (it % report_interval == 0 or it == limit - 1):
                print(
                    f"COUPLE iter={it:04d} mom={final_momentum:.3e} "
                    f"mass={final_mass:.3e} update={final_update:.3e} "
                    f"q_hat={q_hat:.5f} "
                    f"dP={self.delta_p_global:.5e}"
                )
            if (
                self.couple_flow_control
                and self.couple_flow_control_interval > 0
                and (it + 1) % self.couple_flow_control_interval == 0
            ):
                self._relax_delta_p_to_flow(q_hat)
            if final_update < self.tol and final_mass < 5.0 * self.tol and final_momentum < 5.0 * self.tol:
                converged = True
                break
        return SolveLog(
            method="COUPLE",
            converged=converged,
            iterations=it + 1,
            final_momentum=final_momentum,
            final_mass=final_mass,
            final_update=final_update,
            final_q_hat=self.outlet_flow_rate_hat(),
            delta_p=self.delta_p_global,
        )

    def _output_stem(self, prefix: str | None = None) -> str:
        if prefix:
            return prefix
        if self.last_solve_method == "simple":
            return "simple"
        if self.last_solve_method == "couple" and self.pressure_solver == "gmg":
            return "couple_gmg"
        if self.last_solve_method:
            return self.last_solve_method
        return "couple_gmg" if self.pressure_solver == "gmg" else "flow"

    def export_flow_field(self, output_dir: str | Path = "surrogate_debug_outputs", prefix: str | None = None) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = self._output_stem(prefix)
        file_path = output_path / f"{stem}_3d_flow_field.npz"
        history_iteration = np.asarray([item["iteration"] for item in self.iteration_history], dtype=float)
        history_momentum = np.asarray([item["momentum"] for item in self.iteration_history], dtype=float)
        history_continuity = np.asarray([item["continuity"] for item in self.iteration_history], dtype=float)
        history_update = np.asarray([item["update"] for item in self.iteration_history], dtype=float)
        history_q_hat = np.asarray([item["q_hat"] for item in self.iteration_history], dtype=float)
        history_delta_p = np.asarray([item["delta_p"] for item in self.iteration_history], dtype=float)
        history_nut_max = np.asarray([item.get("nut_max", 0.0) for item in self.iteration_history], dtype=float)
        history_nut_mean = np.asarray([item.get("nut_mean", 0.0) for item in self.iteration_history], dtype=float)
        np.savez_compressed(
            file_path,
            R=self.R.detach().cpu().numpy(),
            Theta=self.Theta.detach().cpu().numpy(),
            Z=self.Z.detach().cpu().numpy(),
            phi=self.phi.detach().cpu().numpy(),
            UR=torch.zeros_like(self.UT).detach().cpu().numpy(),
            UT=self.UT.detach().cpu().numpy(),
            UTheta=self.UT.detach().cpu().numpy(),
            UZ=self.Uz.detach().cpu().numpy(),
            P=self.P.detach().cpu().numpy(),
            u_r=torch.zeros_like(self.UT).detach().cpu().numpy(),
            u_theta=(self.UT * self.u_omega).detach().cpu().numpy(),
            u_z=(self.Uz * self.u_zo).detach().cpu().numpy(),
            p=(self.P * self.P0).detach().cpu().numpy(),
            history_iteration=history_iteration,
            history_momentum=history_momentum,
            history_continuity=history_continuity,
            history_update=history_update,
            history_q_hat=history_q_hat,
            history_delta_p=history_delta_p,
            history_nut_max=history_nut_max,
            history_nut_mean=history_nut_mean,
        )
        return file_path

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): BladeCalc._json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [BladeCalc._json_ready(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _condition_parameters_payload(
        self,
        *,
        case_id: str,
        flow_path: Path,
        shape_path: Path,
        visualizations: dict[str, Path],
        log: SolveLog | None,
    ) -> dict[str, Any]:
        result = {
            "method": self.last_solve_method or self._output_stem(),
            "converged": bool(log.converged) if log is not None else None,
            "iterations": int(log.iterations) if log is not None else len(self.iteration_history),
            "final_momentum": float(log.final_momentum) if log is not None else self.momentum_error(),
            "final_mass": float(log.final_mass) if log is not None else self.continuity_error(),
            "final_update": float(log.final_update) if log is not None else None,
            "final_q_hat": float(log.final_q_hat) if log is not None else self.outlet_flow_rate_hat(),
            "delta_p": float(log.delta_p) if log is not None else self.delta_p_global,
        }
        return {
            "schema_version": 1,
            "case_id": case_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": {
                "flow_field": str(flow_path),
                "shape_parameters": str(shape_path),
                **{name: str(path) for name, path in visualizations.items()},
            },
            "condition_parameters": {
                "rh": self.rh,
                "rs": self.rs,
                "h": self.h,
                "z0": self.z0,
                "mu": self.mu,
                "rho": self.rho,
                "nu": self.nu,
                "omega": self.omega,
                "qv": self.qv,
                "n_blade": self.n_blade,
                "absolute_frame": self.absolute_frame,
                "delta_p_initial_or_current": self.delta_p_global,
            },
            "dimensionless_parameters": {
                "u_omega": self.u_omega,
                "u_zo": self.u_zo,
                "P0": self.P0,
                "Re_omega": self.Re_omega,
                "Eu_omega": self.Eu_omega,
                "Lambda": self.Lambda,
                "Ku": self.Ku,
                "delta": self.delta,
                "G_star": self.G_star,
                "qv_passage": self.qv_passage,
                "qv_hat": self.qv_hat,
            },
            "solver_parameters": {
                "n": self.n,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "u_relax": self.u_relax,
                "p_relax": self.p_relax,
                "pressure_solver": self.pressure_solver,
                "pressure_max_inner": self.pressure_max_inner,
                "pressure_tol": self.pressure_tol,
                "pseudo_dt": self.pseudo_dt,
                "local_pseudo_dt": self.local_pseudo_dt,
                "pseudo_cfl": self.pseudo_cfl,
                "rans_model": self.rans_model,
                "rans_mixing_length": self.rans_mixing_length,
                "rans_nut_max_ratio": self.rans_nut_max_ratio,
                "couple_relax": self.couple_relax,
                "momentum_sweeps": self.momentum_sweeps,
                "momentum_solver_relax": self.momentum_solver_relax,
                "gmg_levels": self.gmg_levels,
                "gmg_cycles": self.gmg_cycles,
                "couple_pressure_sweeps": self.couple_pressure_sweeps,
                "couple_pressure_interval": self.couple_pressure_interval,
                "pressure_projection_relax": self.pressure_projection_relax,
            },
            "result_summary": result,
            "iteration_history": self.iteration_history,
        }

    def _shape_parameters_payload(self, *, case_id: str, condition_path: Path, flow_path: Path) -> dict[str, Any]:
        boundary_meta = self.boundary.metadata if self.boundary is not None else None
        mask_points = int(self.blade_mask.detach().cpu().sum().item()) if self.blade_mask is not None else 0
        return {
            "schema_version": 1,
            "case_id": case_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": {
                "condition_parameters": str(condition_path),
                "flow_field": str(flow_path),
                "blade_params": self.blade_params,
            },
            "shape_parameters": {
                "blade_params": self.blade_params,
                "blade_params_is_path": self.blade_params is not None,
                "n_blade": self.n_blade,
                "rh": self.rh,
                "rs": self.rs,
                "h": self.h,
                "z0": self.z0,
                "theta0": self.theta0,
                "delta_r": self.delta_r,
            },
            "ibm_parameters": {
                "ibm_C": self.ibm_C,
                "ibm_epsilon": self.ibm_epsilon,
                "ibm_hard_phi": self.ibm_hard_phi,
                "blade_mask_points": mask_points,
            },
            "blade_boundary_metadata": boundary_meta,
        }

    def export_case_parameters(
        self,
        case_dir: str | Path,
        *,
        case_id: str,
        flow_path: Path,
        visualizations: dict[str, Path] | None = None,
        log: SolveLog | None = None,
    ) -> tuple[Path, Path]:
        output_path = Path(case_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        visualizations = dict(visualizations or {})
        condition_path = output_path / "condition_parameters.json"
        shape_path = output_path / "shape_parameters.json"
        condition_payload = self._condition_parameters_payload(
            case_id=case_id,
            flow_path=flow_path,
            shape_path=shape_path,
            visualizations=visualizations,
            log=log,
        )
        shape_payload = self._shape_parameters_payload(
            case_id=case_id,
            condition_path=condition_path,
            flow_path=flow_path,
        )
        condition_path.write_text(
            json.dumps(self._json_ready(condition_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shape_path.write_text(
            json.dumps(self._json_ready(shape_payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return condition_path, shape_path

    def export_dataset_case(
        self,
        output_root: str | Path = "generated_flow_cases",
        *,
        case_name: str | None = None,
        prefix: str | None = None,
        log: SolveLog | None = None,
        save_visualizations: bool = True,
        plot_3d: bool = False,
        show: bool = False,
    ) -> dict[str, Path]:
        stem = self._output_stem(prefix)
        case_id = case_name or f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        case_dir = Path(output_root) / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        flow_path = self.export_flow_field(case_dir, prefix=case_id)
        visualizations: dict[str, Path] = {}
        if save_visualizations:
            momentum_path, continuity_path, combined_path = self.plot_convergence(case_dir, prefix=case_id)
            slices_path = self.plot_flow_slices(case_dir, prefix=case_id)
            physical_path = self.plot_physical_flow_spans(
                spans=(0.2, 0.5, 0.8),
                show=show,
                save_path=case_dir / f"{case_id}_physical_spans.png",
            )
            visualizations.update(
                {
                    "momentum_residual": momentum_path,
                    "continuity_residual": continuity_path,
                    "convergence": combined_path,
                    "flow_slices": slices_path,
                    "physical_spans": physical_path,
                }
            )
            if plot_3d:
                stream_path = case_dir / f"{case_id}_3d_streamlines.png"
                info = self.plot_3d_streamlines(show=show, save_path=stream_path)
                if info.get("saved", False):
                    visualizations["streamlines_3d"] = stream_path

        condition_path, shape_path = self.export_case_parameters(
            case_dir,
            case_id=case_id,
            flow_path=flow_path,
            visualizations=visualizations,
            log=log,
        )
        return {
            "case_dir": case_dir,
            "flow_field": flow_path,
            "condition_parameters": condition_path,
            "shape_parameters": shape_path,
            **visualizations,
        }

    def plot_convergence(
        self,
        output_dir: str | Path = "surrogate_debug_outputs",
        prefix: str | None = None,
    ) -> tuple[Path, Path, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if not self.iteration_history:
            raise RuntimeError("No iteration history to plot. Run solve() first.")

        iterations = np.asarray([item["iteration"] for item in self.iteration_history], dtype=float)
        momentum = np.asarray([item["momentum"] for item in self.iteration_history], dtype=float)
        continuity = np.asarray([item["continuity"] for item in self.iteration_history], dtype=float)
        momentum = np.clip(momentum, 1e-30, None)
        continuity = np.clip(continuity, 1e-30, None)

        stem = self._output_stem(prefix)
        title = stem.replace("_", "-").upper()
        momentum_path = output_path / f"{stem}_momentum_residual.png"
        continuity_path = output_path / f"{stem}_continuity_residual.png"
        combined_path = output_path / f"{stem}_convergence.png"

        plt.figure(figsize=(7, 4))
        plt.semilogy(iterations, momentum, color="#335c81", linewidth=1.8)
        plt.xlabel("Iteration")
        plt.ylabel("Momentum residual")
        plt.title(f"{title} Momentum Residual")
        plt.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        plt.savefig(momentum_path, dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.semilogy(iterations, continuity, color="#b35c2e", linewidth=1.8)
        plt.xlabel("Iteration")
        plt.ylabel("Continuity residual")
        plt.title(f"{title} Continuity Residual")
        plt.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        plt.savefig(continuity_path, dpi=180)
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        axes[0].semilogy(iterations, momentum, color="#335c81", linewidth=1.8)
        axes[0].set_ylabel("Momentum residual")
        axes[0].grid(True, which="both", alpha=0.25)
        axes[1].semilogy(iterations, continuity, color="#b35c2e", linewidth=1.8)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Continuity residual")
        axes[1].grid(True, which="both", alpha=0.25)
        fig.suptitle(f"{title} Convergence")
        fig.tight_layout()
        fig.savefig(combined_path, dpi=180)
        plt.close(fig)
        return momentum_path, continuity_path, combined_path

    def plot_flow_slices(
        self,
        output_dir: str | Path = "surrogate_debug_outputs",
        spans: tuple[float, ...] = (0.2, 0.5, 0.8),
        prefix: str | None = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = self._output_stem(prefix)
        file_path = output_path / f"{stem}_flow_slices.png"

        fig, axes = plt.subplots(len(spans), 3, figsize=(12, 3.6 * len(spans)), squeeze=False)
        fields = [
            (self.UT.detach().cpu().numpy() * self.u_omega, "u_theta"),
            (self.Uz.detach().cpu().numpy() * self.u_zo, "u_z"),
            (self.P.detach().cpu().numpy() * self.P0, "p"),
        ]
        for row, span in enumerate(spans):
            i = int(np.clip(round(span * (self.n - 1)), 0, self.n - 1))
            for col, (field, name) in enumerate(fields):
                ax = axes[row, col]
                image = ax.imshow(field[i, :, :].T, origin="lower", aspect="auto", cmap="cividis")
                ax.set_title(f"{name} @ span={span:.2f}")
                ax.set_xlabel("Theta index")
                ax.set_ylabel("Z index")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(file_path, dpi=180)
        plt.close(fig)
        return file_path

    @staticmethod
    def _span_to_index(span: float, n: int) -> int:
        span = float(np.clip(span, 0.0, 1.0))
        return int(round(span * (n - 1)))

    @staticmethod
    def _field_stats_np(field: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        values = np.asarray(field)[np.asarray(mask, dtype=bool)]
        if values.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.mean(values)), float(np.min(values)), float(np.max(values))

    def _blade_mask_numpy(self) -> np.ndarray:
        if self.blade_mask is not None:
            return self.blade_mask.detach().cpu().numpy().astype(bool)
        return (self.phi.detach().cpu().numpy() <= self.ibm_hard_phi).astype(bool)

    def _physical_field_arrays(self) -> dict[str, np.ndarray]:
        return {
            "UR": np.zeros(self.r_hatC.shape, dtype=np.float32),
            "UT": (self.UT * self.u_omega).detach().cpu().numpy(),
            "UZ": (self.Uz * self.u_zo).detach().cpu().numpy(),
            "P": (self.P * self.P0).detach().cpu().numpy(),
        }

    def plot_physical_flow_spans(
        self,
        spans: Sequence[float] = (0.2, 0.5, 0.8),
        *,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> Path:
        fields = self._physical_field_arrays()
        blade_mask_3d = self._blade_mask_numpy()
        fluid_mask_3d = ~blade_mask_3d
        n = self.n

        print("\n========== DataGenerator Flow Post Check ==========")
        print(f"grid n={n}, method={self.last_solve_method or self._output_stem()}, pressure_solver={self.pressure_solver}")
        print(f"outlet q_hat={self.outlet_flow_rate_hat():.6g}, delta_p={self.delta_p_global:.6g}")
        print("span statistics are computed in physical SI units over fluid cells.")
        for span in spans:
            r_index = self._span_to_index(float(span), n)
            fluid = fluid_mask_3d[r_index]
            ur_stats = self._field_stats_np(fields["UR"][r_index], fluid)
            ut_stats = self._field_stats_np(fields["UT"][r_index], fluid)
            uz_stats = self._field_stats_np(fields["UZ"][r_index], fluid)
            p_stats = self._field_stats_np(fields["P"][r_index], fluid)
            print(f"span={span:.2f} (i={r_index})")
            print(f"  UR [mean/min/max] = {ur_stats[0]:.6g} / {ur_stats[1]:.6g} / {ur_stats[2]:.6g}")
            print(f"  UT [mean/min/max] = {ut_stats[0]:.6g} / {ut_stats[1]:.6g} / {ut_stats[2]:.6g}")
            print(f"  UZ [mean/min/max] = {uz_stats[0]:.6g} / {uz_stats[1]:.6g} / {uz_stats[2]:.6g}")
            print(f"  P  [mean/min/max] = {p_stats[0]:.6g} / {p_stats[1]:.6g} / {p_stats[2]:.6g}")
        print("===================================================\n")

        fig, axes = plt.subplots(len(spans), 4, figsize=(18, 3.6 * len(spans)), squeeze=False)
        field_names = ["UR", "UT", "UZ", "P"]
        cmaps = {"UR": "coolwarm", "UT": "coolwarm", "UZ": "viridis", "P": "plasma"}
        for row, span in enumerate(spans):
            r_index = self._span_to_index(float(span), n)
            blade_mask = blade_mask_3d[r_index].T
            for col, name in enumerate(field_names):
                data = np.ma.array(fields[name][r_index].T, mask=blade_mask)
                image = axes[row, col].imshow(data, origin="lower", aspect="auto", cmap=cmaps[name])
                if np.any(blade_mask):
                    axes[row, col].contour(blade_mask.astype(float), levels=[0.5], colors="k", linewidths=0.8)
                axes[row, col].set_title(f"{name} @ span={float(span):.2f} (physical)")
                axes[row, col].set_xlabel("Theta index")
                axes[row, col].set_ylabel("Z index")
                fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

        fig.tight_layout()
        if save_path is None:
            save_path = Path("generated_flow_cases") / f"{self._output_stem()}_physical_spans.png"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path

    def _trace_streamline_cylindrical(
        self,
        *,
        fields_phy: dict[str, np.ndarray],
        phi_field: np.ndarray,
        seed_r: float,
        seed_theta: float,
        seed_z: float,
        max_steps: int = 500,
        step_scale: float = 0.75,
        phi_stop: float = 0.25,
    ) -> np.ndarray:
        from SurrogateModelingUtils import interpolate_field_periodic

        dr_cell = self.delta_r / max(self.n - 1, 1)
        dz_cell = self.h / max(self.n - 1, 1)
        dtheta_cell = self.rh * self.theta0 / max(self.n - 1, 1)
        step_length = step_scale * min(dr_cell, dz_cell, dtheta_cell)

        def eval_state(r_value: float, theta_value: float, z_value: float):
            r_norm = (r_value - self.rh) / self.delta_r
            theta_norm = (theta_value % self.theta0) / self.theta0
            z_norm = (z_value - self.z0) / self.h
            phi_value = interpolate_field_periodic(phi_field, r_norm, theta_norm, z_norm)
            if not np.isfinite(phi_value) or phi_value < phi_stop:
                return None
            ur = interpolate_field_periodic(fields_phy["UR"], r_norm, theta_norm, z_norm)
            ut = interpolate_field_periodic(fields_phy["UT"], r_norm, theta_norm, z_norm)
            uz = interpolate_field_periodic(fields_phy["UZ"], r_norm, theta_norm, z_norm)
            if not np.isfinite(ur + ut + uz):
                return None
            speed = float(np.sqrt(ur**2 + ut**2 + uz**2))
            if speed < 1e-10:
                return None
            return np.array([ur, ut / max(r_value, 1e-10), uz], dtype=float), speed

        state = np.array([seed_r, seed_theta, seed_z], dtype=float)
        points: list[np.ndarray] = []
        for _ in range(max_steps):
            if state[0] < self.rh or state[0] > self.rs:
                break
            if state[2] < self.z0 or state[2] > self.z0 + self.h:
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
            points.append(np.array([state[0] * np.cos(state[1]), state[0] * np.sin(state[1]), state[2]], dtype=float))

        if len(points) < 2:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(points)

    def plot_3d_streamlines(
        self,
        *,
        show: bool = True,
        save_path: str | Path | None = None,
        seed_r_count: int = 10,
        seed_theta_count: int = 10,
        passages_to_plot: int | None = None,
        max_streamline_steps: int = 1000,
        streamline_step_scale: float = 0.9,
        theme: str = "dark",
    ) -> dict[str, Any]:
        try:
            import pyvista as pv
            from SurrogateModelingUtils import make_pyvista_blade_surface_meshes, make_pyvista_passage_grid
        except Exception as exc:
            print(f"3D streamline plot skipped: {exc}")
            return {"saved": False, "streamline_count": 0, "reason": str(exc)}

        bg = "#0E1117" if theme == "dark" else "white"
        blade_color = "#FFB000" if theme == "dark" else "#D55E00"
        plotter = pv.Plotter(off_screen=not show, window_size=(1400, 950))
        plotter.set_background(bg)
        plotter.add_mesh(make_pyvista_passage_grid(self), color="lightgray", opacity=0.22, show_edges=True, line_width=0.6)

        if self.boundary is not None:
            for blade_mesh in make_pyvista_blade_surface_meshes(self.boundary, self):
                plotter.add_mesh(
                    blade_mesh,
                    color=blade_color,
                    show_edges=False,
                    smooth_shading=True,
                    ambient=0.20,
                    diffuse=0.85,
                    specular=0.18,
                )

        theta_ring = np.linspace(0.0, 2.0 * np.pi, 240, dtype=float)
        for radius in [self.rh, self.rs]:
            x_ring = radius * np.cos(theta_ring)
            y_ring = radius * np.sin(theta_ring)
            plotter.add_lines(np.column_stack([x_ring, y_ring, np.full_like(x_ring, self.z0)]), color="steelblue", width=2)
            plotter.add_lines(
                np.column_stack([x_ring, y_ring, np.full_like(x_ring, self.z0 + self.h)]),
                color="seagreen",
                width=2,
            )

        phi_field = self.phi.detach().cpu().numpy()
        fields_phy = self._physical_field_arrays()
        phi_inlet = phi_field[:, :, 1 if self.n > 1 else 0]
        r_coords = self.R[:, 0, 0].detach().cpu().numpy()
        theta_coords = self.Theta[0, :, 0].detach().cpu().numpy()
        z_seed = self.z0 + self.h * float(self.Z[0, 0, 1 if self.n > 1 else 0].item())

        r_candidates = np.linspace(1, max(len(r_coords) - 2, 1), num=max(seed_r_count, 1), dtype=int)
        theta_candidates = np.linspace(1, max(len(theta_coords) - 2, 1), num=max(seed_theta_count, 1), dtype=int)
        base_seeds: list[tuple[float, float, float]] = []
        for i_index in r_candidates:
            for j_index in theta_candidates:
                if phi_inlet[i_index, j_index] > 0.75:
                    seed_r = self.rh + float(r_coords[i_index]) * self.delta_r
                    seed_theta = float(theta_coords[j_index]) * self.theta0
                    base_seeds.append((seed_r, seed_theta, z_seed))

        if passages_to_plot is None:
            passages_to_plot = self.n_blade
        passages_to_plot = max(1, min(int(passages_to_plot), self.n_blade))
        colors = plt.cm.viridis(np.linspace(0.12, 0.95, max(len(base_seeds), 1)))[:, :3]
        tube_radius = 0.0075 * self.delta_r
        streamline_count = 0
        for blade_id in range(passages_to_plot):
            theta_shift = blade_id * self.theta0
            for color, (seed_r, seed_theta, seed_z) in zip(colors, base_seeds):
                streamline = self._trace_streamline_cylindrical(
                    fields_phy=fields_phy,
                    phi_field=phi_field,
                    seed_r=seed_r,
                    seed_theta=seed_theta + theta_shift,
                    seed_z=seed_z,
                    max_steps=max_streamline_steps,
                    step_scale=streamline_step_scale,
                )
                if streamline.shape[0] >= 2:
                    plotter.add_mesh(
                        pv.Spline(streamline, max(streamline.shape[0], 2)).tube(radius=tube_radius),
                        color=tuple(float(c) for c in color),
                        smooth_shading=True,
                        opacity=0.92,
                    )
                    streamline_count += 1

        plotter.add_text(
            f"DataGenerator Streamlines | n={self.n} | blades={self.n_blade} | solver={self.pressure_solver}",
            position="upper_left",
            font_size=10,
            color="white" if theme == "dark" else "black",
        )
        plotter.add_axes()
        plotter.camera_position = "iso"
        plotter.camera.zoom(1.18)

        saved = save_path is not None
        if show and save_path is not None:
            plotter.show(screenshot=str(save_path))
        elif show:
            plotter.show()
        else:
            if save_path is not None:
                plotter.screenshot(str(save_path))
            plotter.close()
        return {"saved": saved, "streamline_count": streamline_count, "renderer": "pyvista"}

    def post(self) -> None:
        r = np.linspace(0.0, 1.0, self.n)
        theta = np.linspace(0.0, 1.0, self.n)
        rr, tt = np.meshgrid(r, theta, indexing="ij")

        r_phys = self.rh + rr * (self.rs - self.rh)
        theta_phys = tt * self.theta0
        x = r_phys * np.cos(theta_phys)
        y = r_phys * np.sin(theta_phys)

        k_mid = self.n // 2
        ut = self.UT[:, :, k_mid].detach().cpu().numpy() * self.u_omega
        uz = self.Uz[:, :, k_mid].detach().cpu().numpy() * self.u_zo
        p = self.P[:, :, k_mid].detach().cpu().numpy() * self.P0

        for data, title, label in [
            (ut, "Tangential Velocity", "ut / m s-1"),
            (uz, "Axial Velocity", "uz / m s-1"),
            (p, "Pressure", "p / Pa"),
        ]:
            plt.figure(figsize=(6, 6))
            plt.pcolormesh(x, y, data, shading="auto", cmap="cividis")
            plt.colorbar(label=label)
            plt.title(f"{title} (Physical Space)")
            plt.axis("equal")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

    def plot_blade_boundary(self, span: float = 0.5) -> None:
        if self.blade_mask is None or self.blade_distance is None:
            print("No blade boundary is attached.")
            return
        r_index = int(self.n * span) if span != 1.0 else -1
        r_index = int(np.clip(r_index, 0, self.n - 1))
        mask_slice = self.blade_mask[r_index, :, :].detach().cpu().numpy().T
        dist_slice = self.blade_distance[r_index, :, :].detach().cpu().numpy().T
        phi_slice = self.phi[r_index, :, :].detach().cpu().numpy().T

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].imshow(mask_slice, origin="lower", aspect="auto", cmap="gray_r")
        axes[0].set_title(f"Blade Mask @ i={r_index} (span={span})")
        axes[0].set_xlabel("Theta index j")
        axes[0].set_ylabel("Z index k")

        image = axes[1].imshow(dist_slice, origin="lower", aspect="auto", cmap="coolwarm")
        axes[1].set_title(f"Distance Function @ i={r_index} (span={span})")
        axes[1].set_xlabel("Theta index j")
        axes[1].set_ylabel("Z index k")
        fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

        image = axes[2].imshow(phi_slice, origin="lower", aspect="auto", cmap="GnBu")
        axes[2].set_title(f"Phi Mask @ i={r_index} (span={span})")
        axes[2].set_xlabel("Theta index j")
        axes[2].set_ylabel("Z index k")
        fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_debug_outputs(solver: BladeCalc, output_dir: Path, prefix: str, log: SolveLog | None = None) -> None:
    paths = solver.export_dataset_case(
        output_root=output_dir,
        prefix=prefix,
        log=log,
        save_visualizations=True,
        plot_3d=False,
        show=False,
    )
    print(f"Case directory saved to: {paths['case_dir']}")
    print(f"Flow field saved to: {paths['flow_field']}")
    print(f"Condition parameters saved to: {paths['condition_parameters']}")
    print(f"Shape parameters saved to: {paths['shape_parameters']}")
    if "physical_spans" in paths:
        print(f"Physical span plot saved to: {paths['physical_spans']}")


def run_simple_main() -> SolveLog:
    seed_everything(10492)
    output_dir = Path("generated_flow_cases")
    solver = BladeCalc(
        n=48,
        rh=0.0605,
        rs=0.08,
        h=0.125,
        mu=0.006,
        rho=10650,
        omega=-420 * np.pi / 60,
        qv=0.16,
        n_blade=6,
        max_iter=480,
        tol=1e-4,
        pressure_solver="gmg",
        pressure_max_inner=80,
        pressure_tol=1e-7,
        device="cuda",
        blade_params="../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc/blade_params.json",
        delta_p_initial=0.1,
        pseudo_dt=0.001,
        rans_model="mixing_length",
        rans_mixing_length=0.03,
        rans_nut_max_ratio=20.0,
        local_pseudo_dt=True,
        pseudo_cfl=1.5,
    )
    solver.u_relax = 0.45
    solver.p_relax = 0.25
    solver.momentum_sweeps = 5
    solver.momentum_solver_relax = 0.8
    solver.gmg_cycles = 8
    solver.gmg_levels = 5
    solver.rans_smoothing_steps = 2
    log = solver.solve(method="simple", max_outer=8, report_interval=20)
    _write_debug_outputs(solver, output_dir, "simple", log)
    return log


# 主程序用这个GMG COUPLE Pseudo Transient
def run_couple_gmg_main() -> SolveLog:
    seed_everything(10492)
    output_dir = Path("generated_flow_cases")
    solver = BladeCalc(
        n=256,
        rh=0.0605,
        rs=0.08,
        h=0.125,
        mu=0.006,
        rho=10650,
        omega=-420 * np.pi / 60,
        qv=0.16,
        n_blade=6,
        max_iter=360,
        tol=1e-4,
        pressure_solver="gmg",
        pressure_max_inner=800,
        pressure_tol=1e-7,
        device="cuda",
        blade_params="../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc/blade_params.json",
        delta_p_initial=0.1,
        pseudo_dt=0.0005,
        rans_model="mixing_length",
        rans_mixing_length=0.03,
        rans_nut_max_ratio=20.0,
        local_pseudo_dt=True,
        pseudo_cfl=1.5,
    )
    solver.couple_relax = 0.2
    solver.couple_momentum_update_limit = 0.2
    solver.couple_pressure_velocity_limit = 0.2
    solver.couple_pressure_update_limit = 1.0
    solver.couple_flow_control = False
    solver.couple_flow_control_interval = 20
    solver.couple_flow_control_gain = 0.25
    solver.couple_flow_control_clip = 0.08
    solver.momentum_sweeps = 10
    solver.momentum_solver_relax = 0.8
    solver.couple_pressure_backtracking = True
    solver.couple_pressure_max_momentum_growth = 1.0
    solver.couple_pressure_min_relax = 1e-3
    solver.couple_pressure_interval = 10
    solver.rans_smoothing_steps = 1
    solver.gmg_cycles = 8
    solver.gmg_levels = 6
    solver.couple_pressure_sweeps = 1
    solver.pressure_projection_relax = 0.25
    log = solver.solve(method="couple", report_interval=1)
    _write_debug_outputs(solver, output_dir, "couple_gmg", log)
    return log


def run_main(method: str = "couple") -> SolveLog:
    method = method.lower().replace("-", "_")
    if method in {"simple", "simple_gmg"}:
        return run_simple_main()
    if method in {"couple", "couple_gmg", "gmg"}:
        return run_couple_gmg_main()
    raise ValueError("method must be 'couple' or 'simple'.")


if __name__ == "__main__":
    selected_method = "couple_gmg"   # 主程序生成
    log = run_main(selected_method)
    print(log)
