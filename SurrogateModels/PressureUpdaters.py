from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class PressureUpdater:
    """Pressure-correction and pressure linear-system manager for BladeCalc.

    The flow solver owns geometry, fields, Rhie-Chow fluxes, and boundary
    application. This class owns everything that assembles and solves pressure
    corrections, including Jacobi, BiCGStab, GMG, SIMPLE pressure updates, and
    the pseudo-transient COUPLE pressure projection.
    """

    def __init__(self, flow: Any):
        self.flow = flow

    def _operator(self, p_prime: torch.Tensor, coef: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._operator_for(p_prime, coef, self.flow.blade_mask)

    def _operator_for(
        self,
        p_prime: torch.Tensor,
        coef: dict[str, torch.Tensor],
        blade_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flow = self.flow
        p_prime = p_prime.clone()
        flow._project_theta_periodic(p_prime)
        pn = flow.neighbor(p_prime, "N")
        ps = flow.neighbor(p_prime, "S")
        pt = flow.neighbor(p_prime, "T")
        pb = flow.neighbor(p_prime, "B")
        out = coef["C"] * p_prime - (coef["N"] * pn + coef["S"] * ps + coef["T"] * pt + coef["B"] * pb)
        out[:, :, 0] = p_prime[:, :, 0]
        out[:, :, -1] = p_prime[:, :, -1]
        out[:, -1, :] = out[:, 0, :]
        if blade_mask is not None and blade_mask.shape == p_prime.shape:
            out[blade_mask] = p_prime[blade_mask]
        return out

    def _project(self, x: torch.Tensor, blade_mask: torch.Tensor | None = None) -> torch.Tensor:
        flow = self.flow
        x[:, :, 0] = 0.0
        x[:, :, -1] = 0.0
        flow._project_theta_periodic(x)
        if blade_mask is not None and blade_mask.shape == x.shape:
            x[blade_mask] = 0.0
        return x

    @staticmethod
    def _pool_keep_r(x: torch.Tensor) -> torch.Tensor:
        nr, nt, nz = x.shape
        nt_even = nt - (nt % 2)
        nz_even = nz - (nz % 2)
        x_even = x[:, :nt_even, :nz_even]
        pooled = F.avg_pool2d(x_even.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
        if pooled.shape[1] >= 2:
            pooled[:, -1, :] = pooled[:, 0, :]
        pooled[:, :, 0] = 0.0
        pooled[:, :, -1] = 0.0
        return pooled

    @staticmethod
    def _prolong_keep_r(x: torch.Tensor, target_shape: torch.Size | tuple[int, int, int]) -> torch.Tensor:
        _, nt, nz = target_shape
        out = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        if out.shape[1] < nt:
            out = torch.cat([out, out[:, : nt - out.shape[1], :]], dim=1)
        if out.shape[2] < nz:
            out = torch.cat([out, out[:, :, -1:].expand(-1, -1, nz - out.shape[2])], dim=2)
        out = out[:, :nt, :nz]
        if out.shape[1] >= 2:
            out[:, -1, :] = out[:, 0, :]
        out[:, :, 0] = 0.0
        out[:, :, -1] = 0.0
        return out

    def coefficients(
        self,
        inv_theta: torch.Tensor,
        inv_z: torch.Tensor,
        beta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flow = self.flow
        dtheta_c = flow.dA * inv_theta
        dz_c = flow.dA * inv_z
        dtheta_n = 0.5 * (dtheta_c + flow.neighbor(dtheta_c, "N"))
        dtheta_s = 0.5 * (dtheta_c + flow.neighbor(dtheta_c, "S"))
        dz_t = 0.5 * (dz_c + flow.neighbor(dz_c, "T"))
        dz_b = 0.5 * (dz_c + flow.neighbor(dz_c, "B"))
        phi_n, phi_s, phi_t, phi_b = flow._face_opening()

        alpha_n = phi_n * (flow.K_theta_C**2) * flow.Eu_omega * dtheta_n * flow.theta_area / flow.dTheta
        alpha_s = phi_s * (flow.K_theta_C**2) * flow.Eu_omega * dtheta_s * flow.theta_area / flow.dTheta
        alpha_t = phi_t * (flow.Lambda**2) * flow.Eu_omega * dz_t * flow.z_area / flow.dZ
        alpha_b = phi_b * (flow.Lambda**2) * flow.Eu_omega * dz_b * flow.z_area / flow.dZ
        alpha_c = alpha_n + alpha_s + alpha_t + alpha_b + 1e-12

        for field in (alpha_n, alpha_s, alpha_t, alpha_b, alpha_c, beta):
            field[:, :, 0] = 0.0
            field[:, :, -1] = 0.0
            field[:, -1, :] = field[:, 0, :]
        alpha_c[:, :, 0] = 1.0
        alpha_c[:, :, -1] = 1.0

        return {
            "C": alpha_c,
            "N": alpha_n,
            "S": alpha_s,
            "T": alpha_t,
            "B": alpha_b,
        }

    def correction_gradient(self, p_prime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow = self.flow
        gt_prime = flow.dA * flow.Eu_omega * flow.K_theta_C * flow._pressure_gradient_theta(p_prime)
        gz_prime = (
            flow.dA
            * flow.Eu_omega
            * (flow.Lambda / max(flow.Ku, 1e-12))
            * flow._pressure_gradient_z(p_prime)
        )
        return gt_prime, gz_prime

    def solve_bicgstab(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        x = flow.P_prime.clone()
        x[:, :, 0] = 0.0
        x[:, :, -1] = 0.0
        flow._project_theta_periodic(x)
        b = beta.clone()
        b[:, :, 0] = 0.0
        b[:, :, -1] = 0.0
        b[:, -1, :] = b[:, 0, :]
        if flow.blade_mask is not None:
            b[flow.blade_mask] = 0.0
            x[flow.blade_mask] = 0.0

        r = b - self._operator(x, coef)
        r_hat = r.clone()
        rho_old = torch.tensor(1.0, device=flow.device)
        alpha = torch.tensor(1.0, device=flow.device)
        omega = torch.tensor(1.0, device=flow.device)
        v = torch.zeros_like(x)
        p = torch.zeros_like(x)

        for _ in range(flow.pressure_max_inner):
            rho_new = torch.sum(r_hat * r)
            if torch.abs(rho_new) < 1e-30 or torch.abs(rho_old) < 1e-30 or torch.abs(omega) < 1e-30:
                break
            beta_k = (rho_new / rho_old) * (alpha / omega)
            p = r + beta_k * (p - omega * v)
            v = self._operator(p, coef)
            den_alpha = torch.sum(r_hat * v)
            if torch.abs(den_alpha) < 1e-30:
                break
            alpha = rho_new / den_alpha
            s = r - alpha * v
            if torch.linalg.vector_norm(s) < flow.pressure_tol:
                x = x + alpha * p
                break
            t = self._operator(s, coef)
            tt = torch.sum(t * t)
            if torch.abs(tt) < 1e-30:
                break
            omega = torch.sum(t * s) / tt
            x = x + alpha * p + omega * s
            x[:, :, 0] = 0.0
            x[:, :, -1] = 0.0
            flow._project_theta_periodic(x)
            if flow.blade_mask is not None:
                x[flow.blade_mask] = 0.0
            r = s - omega * t
            rho_old = rho_new
            if torch.linalg.vector_norm(r) < flow.pressure_tol:
                break

        x[:, :, 0] = 0.0
        x[:, :, -1] = 0.0
        flow._project_theta_periodic(x)
        if flow.blade_mask is not None:
            x[flow.blade_mask] = 0.0
        return x

    def solve_jacobi(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        x = flow.P_prime.clone()
        x[:, :, 0] = 0.0
        x[:, :, -1] = 0.0
        flow._project_theta_periodic(x)
        beta = beta.clone()
        beta[:, :, 0] = 0.0
        beta[:, :, -1] = 0.0
        beta[:, -1, :] = beta[:, 0, :]

        for _ in range(flow.pressure_max_inner):
            rhs = (
                coef["N"] * flow.neighbor(x, "N")
                + coef["S"] * flow.neighbor(x, "S")
                + coef["T"] * flow.neighbor(x, "T")
                + coef["B"] * flow.neighbor(x, "B")
                + beta
            )
            x_new = rhs / torch.clamp(coef["C"], min=1e-12)
            x_new[:, :, 0] = 0.0
            x_new[:, :, -1] = 0.0
            flow._project_theta_periodic(x_new)
            if flow.blade_mask is not None:
                x_new[flow.blade_mask] = 0.0
            err = torch.mean(torch.abs(x_new - x))
            x = 0.75 * x_new + 0.25 * x
            if err < flow.pressure_tol:
                break
        return x

    def _coarsen_coef(self, coef: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coarse = {name: self._pool_keep_r(value) for name, value in coef.items()}
        for name in ("N", "S", "T", "B"):
            coarse[name][:, :, 0] = 0.0
            coarse[name][:, :, -1] = 0.0
            coarse[name][:, -1, :] = coarse[name][:, 0, :]
        coarse["C"] = torch.clamp(
            coarse["N"] + coarse["S"] + coarse["T"] + coarse["B"],
            min=1e-12,
        )
        coarse["C"][:, :, 0] = 1.0
        coarse["C"][:, :, -1] = 1.0
        if coarse["C"].shape[1] >= 2:
            coarse["C"][:, -1, :] = coarse["C"][:, 0, :]
        return coarse

    def _smooth_gmg(
        self,
        coef: dict[str, torch.Tensor],
        beta: torch.Tensor,
        x: torch.Tensor,
        steps: int,
        omega: float,
    ) -> torch.Tensor:
        for _ in range(steps):
            residual = beta - self._operator_for(x, coef, None)
            x = x + omega * residual / torch.clamp(coef["C"], min=1e-12)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            self._project(x)
        return x

    def _v_cycle(
        self,
        coef_levels: list[dict[str, torch.Tensor]],
        level: int,
        beta: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        flow = self.flow
        coef = coef_levels[level]
        if level == len(coef_levels) - 1 or min(x.shape[1], x.shape[2]) <= 4:
            return self._smooth_gmg(coef, beta, x, 30, flow.gmg_omega)

        x = self._smooth_gmg(coef, beta, x, flow.gmg_pre_smooth, flow.gmg_omega)
        residual = beta - self._operator_for(x, coef, None)
        residual_c = self._pool_keep_r(residual)
        error_c = torch.zeros_like(residual_c)
        error_c = self._v_cycle(coef_levels, level + 1, residual_c, error_c)
        x = x + self._prolong_keep_r(error_c, x.shape)
        self._project(x)
        x = self._smooth_gmg(coef, beta, x, flow.gmg_post_smooth, flow.gmg_omega)
        return x

    def solve_gmg(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        beta = beta.clone()
        beta[:, :, 0] = 0.0
        beta[:, :, -1] = 0.0
        if beta.shape[1] >= 2:
            beta[:, -1, :] = beta[:, 0, :]
        if flow.blade_mask is not None and flow.blade_mask.shape == beta.shape:
            beta[flow.blade_mask] = 0.0

        coef_levels = [coef]
        for _ in range(1, flow.gmg_levels):
            last_shape = coef_levels[-1]["C"].shape
            if min(last_shape[1], last_shape[2]) <= 4:
                break
            coef_levels.append(self._coarsen_coef(coef_levels[-1]))

        x = flow.P_prime.clone()
        if x.shape != beta.shape:
            x = torch.zeros_like(beta)
        x = self._project(x, flow.blade_mask)

        rhs_norm = torch.linalg.vector_norm(beta).item()
        for _ in range(flow.gmg_cycles):
            x = self._v_cycle(coef_levels, 0, beta, x)
            residual = beta - self._operator_for(x, coef, flow.blade_mask)
            res_norm = torch.linalg.vector_norm(residual).item()
            if res_norm < flow.pressure_tol * max(rhs_norm, 1.0):
                break
        return self._project(x, flow.blade_mask)

    def solve_system(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        solver_name = self.flow.pressure_solver
        if solver_name == "jacobi":
            return self.solve_jacobi(coef, beta)
        if solver_name == "gmg":
            return self.solve_gmg(coef, beta)
        return self.solve_bicgstab(coef, beta)

    def simple_step(self) -> None:
        flow = self.flow
        inv_theta = flow.phi / torch.clamp(flow.A_theta + (1.0 - flow.phi), min=1e-12)
        inv_z = flow.phi / torch.clamp(flow.A_z + (1.0 - flow.phi), min=1e-12)

        ut_n_face, ut_s_face, uz_t_face, uz_b_face = flow.rhie_chow(
            flow.UT_tilde,
            flow.Uz_tilde,
            flow.P,
            flow.A_theta,
            flow.A_z,
        )
        fn, fs, ft, fb = flow._face_fluxes(ut_n_face, ut_s_face, uz_t_face, uz_b_face)
        phi_n, phi_s, phi_t, phi_b = flow._face_opening()
        fn = fn * phi_n
        fs = fs * phi_s
        ft = ft * phi_t
        fb = fb * phi_b
        beta = -(fn + fs + ft + fb)
        coef = self.coefficients(inv_theta, inv_z, beta)
        p_prime = self.solve_system(coef, beta)
        flow.P_prime = p_prime

        gt_prime, gz_prime = self.correction_gradient(p_prime)
        ut_corr = flow.UT_tilde - inv_theta * gt_prime
        uz_corr = flow.Uz_tilde - inv_z * gz_prime
        flow.UT = flow.UT + flow.u_relax * (ut_corr - flow.UT)
        flow.Uz = flow.Uz + flow.u_relax * (uz_corr - flow.Uz)
        flow.P = flow.P + flow.p_relax * p_prime

    def project(
        self,
        inv_theta: torch.Tensor,
        inv_z: torch.Tensor,
        beta: torch.Tensor,
        relax: float,
        velocity_update_limit: float | None = None,
        pressure_update_limit: float | None = None,
    ) -> torch.Tensor:
        flow = self.flow
        pcoef = self.coefficients(inv_theta, inv_z, beta)
        p_prime = self.solve_system(pcoef, beta)
        gt_prime, gz_prime = self.correction_gradient(p_prime)

        d_ut = -relax * inv_theta * gt_prime
        d_uz = -relax * inv_z * gz_prime
        d_p = relax * p_prime
        scale = min(
            flow._velocity_update_scale(d_ut, d_uz, velocity_update_limit),
            flow._pressure_update_scale(d_p, pressure_update_limit),
        )
        if scale <= 0.0:
            flow.P_prime = torch.zeros_like(p_prime)
            return flow.P_prime

        flow.P_prime = scale * p_prime
        flow.UT = flow.UT + scale * d_ut
        flow.Uz = flow.Uz + scale * d_uz
        flow.P = flow.P + scale * d_p
        flow._apply_boundary()
        return flow.P_prime
