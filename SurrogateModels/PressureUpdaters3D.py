from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class PressureUpdater3D:
    """三维压力修正求解器。

    DataGenerator3D 负责速度、几何、边界、IBM 和动量方程；本类只处理压力修正：

    1. 根据动量方程局部矩阵 A 的逆构造 SIMPLE 压力修正方程；
    2. 支持 Jacobi、BiCGStab 和几何多重网格 GMG；
    3. 在 SIMPLE 中做完整压力步；
    4. 在 COUPLE 中提供带限幅的压力投影。

    3D 柱坐标 SIMPLE 最容易出错的地方，是把速度修正写成直角坐标的 1 / aP 对角近似。Dimensionless Document.pdf 的式 (2.86)
    到 (2.96) 要求先反演局部 [UR, UTheta] 2x2 子块，再用 Schur complement 构造
    压力方程；下面的 invA 字段就是为此准备的。
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
        """计算 A_p p'，其中 A_p 是六点压力修正离散算子。

        系数命名遵循 FVM 六面体习惯：E/W 为径向外/内，N/S 为周向正/负，T/B 为
        轴向出口/入口。入口和出口采用 p'=0 的 Dirichlet 投影，径向壁面通过面系数
        置零实现无压力修正通量。
        """
        flow = self.flow
        x = p_prime.clone()
        self._project(x, blade_mask)
        pe = flow.neighbor(x, "E")
        pw = flow.neighbor(x, "W")
        pn = flow.neighbor(x, "N")
        ps = flow.neighbor(x, "S")
        pt = flow.neighbor(x, "T")
        pb = flow.neighbor(x, "B")
        pne = flow.neighbor(pe, "N")
        pnw = flow.neighbor(pw, "N")
        pse = flow.neighbor(pe, "S")
        psw = flow.neighbor(pw, "S")
        out = (
            coef["C"] * x
            - coef["E"] * pe
            - coef["W"] * pw
            - coef["N"] * pn
            - coef["S"] * ps
            - coef["T"] * pt
            - coef["B"] * pb
            - coef.get("NE", 0.0) * pne
            - coef.get("NW", 0.0) * pnw
            - coef.get("SE", 0.0) * pse
            - coef.get("SW", 0.0) * psw
        )
        out[:, :, 0] = x[:, :, 0]
        out[:, :, -1] = x[:, :, -1]
        flow._project_theta_periodic(out)
        if blade_mask is not None and blade_mask.shape == x.shape:
            out[blade_mask] = x[blade_mask]
        return out

    def _project(self, x: torch.Tensor, blade_mask: torch.Tensor | None = None) -> torch.Tensor:
        """投影压力修正边界。

        原 SIMPLE 压力修正中固定压力边界取 p'=0。当前求解器入口/出口压力由
        delta_p_global 和出口基准压力给定，因此两端 p'=0。径向壁面保留内部值，
        配合 E/W 面系数为 0，形成无通量边界。
        """
        flow = self.flow
        x[:, :, 0] = 0.0
        x[:, :, -1] = 0.0
        flow._project_theta_periodic(x)
        if blade_mask is not None and blade_mask.shape == x.shape:
            x[blade_mask] = 0.0
        return x

    @staticmethod
    def _pool_keep_shape(x: torch.Tensor) -> torch.Tensor:
        """三维 GMG 限制算子。

        径向、周向、轴向都按 2 倍粗化；周向末端仍投影为周期重复点。
        """
        nr, nt, nz = x.shape
        nr_even = nr - (nr % 2)
        nt_even = nt - (nt % 2)
        nz_even = nz - (nz % 2)
        x_even = x[:nr_even, :nt_even, :nz_even]
        pooled = F.avg_pool3d(x_even.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2).squeeze(0).squeeze(0)
        if pooled.shape[1] >= 2:
            pooled[:, -1, :] = pooled[:, 0, :]
        pooled[:, :, 0] = 0.0
        pooled[:, :, -1] = 0.0
        return pooled

    @staticmethod
    def _prolong_keep_shape(x: torch.Tensor, target_shape: torch.Size | tuple[int, int, int]) -> torch.Tensor:
        """三维 GMG 延拓算子，使用简单 piecewise-constant prolongation。"""
        nr, nt, nz = target_shape
        out = x.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        if out.shape[0] < nr:
            out = torch.cat([out, out[-1:, :, :].expand(nr - out.shape[0], -1, -1)], dim=0)
        if out.shape[1] < nt:
            out = torch.cat([out, out[:, : nt - out.shape[1], :]], dim=1)
        if out.shape[2] < nz:
            out = torch.cat([out, out[:, :, -1:].expand(-1, -1, nz - out.shape[2])], dim=2)
        out = out[:nr, :nt, :nz]
        if out.shape[1] >= 2:
            out[:, -1, :] = out[:, 0, :]
        out[:, :, 0] = 0.0
        out[:, :, -1] = 0.0
        return out

    def inverse_momentum_blocks(self) -> dict[str, torch.Tensor]:
        """返回动量局部矩阵 A 的逆块。

        A = [[A11, A12, 0],
             [A21, A22, 0],
             [0,   0,   A33]]

        这是 Dimensionless Document.pdf 式 (2.95) 的代码形式。即便当前默认的
        A12/A21 被强限幅以提高鲁棒性，压力方程仍通过完整 2x2 逆构造，而不是
        简化为 1/A11、1/A22。
        """
        flow = self.flow
        a11 = torch.clamp(flow.A11 + (1.0 - flow.phi), min=1e-12)
        a22 = torch.clamp(flow.A22 + (1.0 - flow.phi), min=1e-12)
        a33 = torch.clamp(flow.A33 + (1.0 - flow.phi), min=1e-12)
        a12 = torch.nan_to_num(flow.A12, nan=0.0, posinf=0.0, neginf=0.0)
        a21 = torch.nan_to_num(flow.A21, nan=0.0, posinf=0.0, neginf=0.0)
        det2 = torch.clamp(a11 * a22 - a12 * a21, min=1e-12)
        return {
            "11": a22 / det2,
            "12": -a12 / det2,
            "21": -a21 / det2,
            "22": a11 / det2,
            "33": 1.0 / a33,
        }

    def coefficients(self, inv_blocks: dict[str, torch.Tensor], beta: torch.Tensor) -> dict[str, torch.Tensor]:
        """组装压力修正方程的压力修正系数。

        代码采用式 (2.98)-(2.105) 的稳定化实现：
        - 对角贡献 E/W、N/S、T/B 进入主方向模板；
        - R-Theta 交叉导数项显式形成 NE/NW/SE/SW 角点模板；
        - 若角点耦合非零，solve_system() 会避开当前 GMG 平滑近似并使用 BiCGSTAB。
        """
        flow = self.flow
        # SIMPLE 速度修正中，梯度项被 dV * Eu * metric 包装；这里的 d_* 就是
        # 面上速度对压力差的响应强度。
        d_r = flow.dV * inv_blocks["11"] * flow.phi
        d_rt = flow.dV * inv_blocks["12"] * flow.phi
        d_tr = flow.dV * inv_blocks["21"] * flow.phi
        d_t = flow.dV * inv_blocks["22"] * flow.phi
        d_z = flow.dV * inv_blocks["33"] * flow.phi

        d_r_e = 0.5 * (d_r + flow.neighbor(d_r, "E"))
        d_r_w = 0.5 * (d_r + flow.neighbor(d_r, "W"))
        d_rt_e = 0.5 * (d_rt + flow.neighbor(d_rt, "E"))
        d_rt_w = 0.5 * (d_rt + flow.neighbor(d_rt, "W"))
        d_tr_n = 0.5 * (d_tr + flow.neighbor(d_tr, "N"))
        d_tr_s = 0.5 * (d_tr + flow.neighbor(d_tr, "S"))
        d_t_n = 0.5 * (d_t + flow.neighbor(d_t, "N"))
        d_t_s = 0.5 * (d_t + flow.neighbor(d_t, "S"))
        d_z_t = 0.5 * (d_z + flow.neighbor(d_z, "T"))
        d_z_b = 0.5 * (d_z + flow.neighbor(d_z, "B"))

        phi_e, phi_w, phi_n, phi_s, phi_t, phi_b = flow._face_opening()

        # 径向通量来自 1/r d(r UR)/dR，面权重使用 r_face/r_cell。
        radial_e = torch.clamp(flow.r_hat_E / flow.r_hatC, min=0.0)
        radial_w = torch.clamp(flow.r_hat_W / flow.r_hatC, min=0.0)
        k_theta_e = 0.5 * (flow.K_theta_C + flow.neighbor(flow.K_theta_C, "E"))
        k_theta_w = 0.5 * (flow.K_theta_C + flow.neighbor(flow.K_theta_C, "W"))
        alpha_e_direct = phi_e * radial_e * flow.radial_area * flow.Eu_omega * d_r_e / flow.dR
        alpha_w_direct = phi_w * radial_w * flow.radial_area * flow.Eu_omega * d_r_w / flow.dR
        alpha_rt_e = (
            phi_e * radial_e * flow.radial_area * flow.Eu_omega * k_theta_e * d_rt_e / (4.0 * flow.dTheta)
        )
        alpha_rt_w = (
            phi_w * radial_w * flow.radial_area * flow.Eu_omega * k_theta_w * d_rt_w / (4.0 * flow.dTheta)
        )

        # 周向：通量 K_theta * UTheta，压力梯度 Eu*K_theta*dP/dTheta，所以 K_theta^2。
        alpha_n_direct = phi_n * (flow.K_theta_C**2) * flow.theta_area * flow.Eu_omega * d_t_n / flow.dTheta
        alpha_s_direct = phi_s * (flow.K_theta_C**2) * flow.theta_area * flow.Eu_omega * d_t_s / flow.dTheta
        alpha_tr_n = phi_n * flow.K_theta_C * flow.theta_area * flow.Eu_omega * d_tr_n / (4.0 * flow.dR)
        alpha_tr_s = phi_s * flow.K_theta_C * flow.theta_area * flow.Eu_omega * d_tr_s / (4.0 * flow.dR)

        # 轴向：通量 Lambda*Ku*UZ，压力梯度 Eu*(Lambda/Ku)*dP/dZ，Ku 抵消后是 Lambda^2。
        alpha_t = phi_t * (flow.Lambda**2) * flow.z_area * flow.Eu_omega * d_z_t / flow.dZ
        alpha_b = phi_b * (flow.Lambda**2) * flow.z_area * flow.Eu_omega * d_z_b / flow.dZ

        # 边界面无压力修正通量。
        alpha_e_direct[-1, :, :] = 0.0
        alpha_e_direct[-1, :, :] = 0.0
        alpha_w_direct[0, :, :] = 0.0
        alpha_rt_e[-1, :, :] = 0.0
        alpha_rt_w[0, :, :] = 0.0
        alpha_t[:, :, -1] = 0.0
        alpha_b[:, :, 0] = 0.0

        alpha_e = alpha_e_direct + alpha_tr_n - alpha_tr_s
        alpha_w = alpha_w_direct - alpha_tr_n + alpha_tr_s
        alpha_n = alpha_rt_e - alpha_rt_w + alpha_n_direct
        alpha_s = -alpha_rt_e + alpha_rt_w + alpha_s_direct
        alpha_ne = alpha_rt_e + alpha_tr_n
        alpha_se = -alpha_rt_e - alpha_tr_s
        alpha_nw = -alpha_rt_w - alpha_tr_n
        alpha_sw = alpha_rt_w + alpha_tr_s

        coeff_fields = (
            alpha_e,
            alpha_w,
            alpha_n,
            alpha_s,
            alpha_t,
            alpha_b,
            alpha_ne,
            alpha_nw,
            alpha_se,
            alpha_sw,
        )
        for field in (*coeff_fields, beta):
            torch.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0, out=field)
            flow._project_theta_periodic(field)
            field[:, :, 0] = 0.0
            field[:, :, -1] = 0.0
        alpha_c = alpha_e + alpha_w + alpha_n + alpha_s + alpha_t + alpha_b + 1e-12
        alpha_c[:, :, 0] = 1.0
        alpha_c[:, :, -1] = 1.0
        flow._project_theta_periodic(alpha_c)

        return {
            "C": alpha_c,
            "E": alpha_e,
            "W": alpha_w,
            "N": alpha_n,
            "S": alpha_s,
            "T": alpha_t,
            "B": alpha_b,
            "NE": alpha_ne,
            "NW": alpha_nw,
            "SE": alpha_se,
            "SW": alpha_sw,
        }

    def correction_gradient(self, p_prime: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 dV*G(P')，即式 (2.88) 的离散梯度向量。"""
        flow = self.flow
        g_r = flow.dV * flow.Eu_omega * flow._pressure_gradient_r(p_prime)
        g_t = flow.dV * flow.Eu_omega * flow.K_theta_C * flow._pressure_gradient_theta(p_prime)
        g_z = flow.dV * flow.Eu_omega * (flow.Lambda / max(flow.Ku, 1e-12)) * flow._pressure_gradient_z(p_prime)
        return g_r, g_t, g_z

    def velocity_correction(
        self,
        p_prime: torch.Tensor,
        inv_blocks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """根据 A^{-1}G(P') 计算速度修正量。"""
        inv_blocks = inv_blocks if inv_blocks is not None else self.inverse_momentum_blocks()
        g_r, g_t, g_z = self.correction_gradient(p_prime)
        d_ur = -(inv_blocks["11"] * g_r + inv_blocks["12"] * g_t)
        d_ut = -(inv_blocks["21"] * g_r + inv_blocks["22"] * g_t)
        d_uz = -(inv_blocks["33"] * g_z)
        return d_ur, d_ut, d_uz

    def solve_jacobi(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        x = self._project(flow.P_prime.clone(), flow.blade_mask)
        b = beta.clone()
        b[:, :, 0] = 0.0
        b[:, :, -1] = 0.0
        flow._project_theta_periodic(b)
        if flow.blade_mask is not None:
            b[flow.blade_mask] = 0.0

        for _ in range(flow.pressure_max_inner):
            rhs = (
                coef["E"] * flow.neighbor(x, "E")
                + coef["W"] * flow.neighbor(x, "W")
                + coef["N"] * flow.neighbor(x, "N")
                + coef["S"] * flow.neighbor(x, "S")
                + coef["T"] * flow.neighbor(x, "T")
                + coef["B"] * flow.neighbor(x, "B")
                + coef.get("NE", 0.0) * flow.neighbor(flow.neighbor(x, "E"), "N")
                + coef.get("NW", 0.0) * flow.neighbor(flow.neighbor(x, "W"), "N")
                + coef.get("SE", 0.0) * flow.neighbor(flow.neighbor(x, "E"), "S")
                + coef.get("SW", 0.0) * flow.neighbor(flow.neighbor(x, "W"), "S")
                + b
            )
            x_new = rhs / torch.clamp(coef["C"], min=1e-12)
            self._project(x_new, flow.blade_mask)
            err = torch.mean(torch.abs(x_new - x))
            x = 0.75 * x_new + 0.25 * x
            if err < flow.pressure_tol:
                break
        return self._project(x, flow.blade_mask)

    def solve_bicgstab(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        x = self._project(flow.P_prime.clone(), flow.blade_mask)
        b = beta.clone()
        b[:, :, 0] = 0.0
        b[:, :, -1] = 0.0
        flow._project_theta_periodic(b)
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
            self._project(x, flow.blade_mask)
            r = s - omega * t
            rho_old = rho_new
            if torch.linalg.vector_norm(r) < flow.pressure_tol:
                break
        return self._project(x, flow.blade_mask)

    def _coarsen_coef(self, coef: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coarse = {name: self._pool_keep_shape(value) for name, value in coef.items()}
        for name in ("E", "W", "N", "S", "T", "B", "NE", "NW", "SE", "SW"):
            coarse[name][:, :, 0] = 0.0
            coarse[name][:, :, -1] = 0.0
            if coarse[name].shape[1] >= 2:
                coarse[name][:, -1, :] = coarse[name][:, 0, :]
        coarse["C"] = torch.clamp(
            coarse["E"] + coarse["W"] + coarse["N"] + coarse["S"] + coarse["T"] + coarse["B"],
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
        if level == len(coef_levels) - 1 or min(x.shape) <= 4:
            return self._smooth_gmg(coef, beta, x, 28, flow.gmg_omega)
        x = self._smooth_gmg(coef, beta, x, flow.gmg_pre_smooth, flow.gmg_omega)
        residual = beta - self._operator_for(x, coef, None)
        residual_c = self._pool_keep_shape(residual)
        error_c = torch.zeros_like(residual_c)
        error_c = self._v_cycle(coef_levels, level + 1, residual_c, error_c)
        x = x + self._prolong_keep_shape(error_c, x.shape)
        self._project(x)
        return self._smooth_gmg(coef, beta, x, flow.gmg_post_smooth, flow.gmg_omega)

    def solve_gmg(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        flow = self.flow
        b = beta.clone()
        b[:, :, 0] = 0.0
        b[:, :, -1] = 0.0
        flow._project_theta_periodic(b)
        if flow.blade_mask is not None and flow.blade_mask.shape == b.shape:
            b[flow.blade_mask] = 0.0

        coef_levels = [coef]
        for _ in range(1, flow.gmg_levels):
            last_shape = coef_levels[-1]["C"].shape
            if min(last_shape) <= 4:
                break
            coef_levels.append(self._coarsen_coef(coef_levels[-1]))

        x = flow.P_prime.clone()
        if x.shape != b.shape:
            x = torch.zeros_like(b)
        x = self._project(x, flow.blade_mask)

        rhs_norm = torch.linalg.vector_norm(b).item()
        for _ in range(flow.gmg_cycles):
            x = self._v_cycle(coef_levels, 0, b, x)
            residual = b - self._operator_for(x, coef, flow.blade_mask)
            res_norm = torch.linalg.vector_norm(residual).item()
            if res_norm < flow.pressure_tol * max(rhs_norm, 1.0):
                break
        return self._project(x, flow.blade_mask)

    def solve_system(self, coef: dict[str, torch.Tensor], beta: torch.Tensor) -> torch.Tensor:
        solver_name = self.flow.pressure_solver
        has_corner_coupling = any(
            name in coef and torch.max(torch.abs(coef[name])).item() > 1e-18
            for name in ("NE", "NW", "SE", "SW")
        )
        if solver_name == "gmg" and has_corner_coupling:
            return self.solve_bicgstab(coef, beta)
        if solver_name == "jacobi":
            return self.solve_jacobi(coef, beta)
        if solver_name == "gmg":
            return self.solve_gmg(coef, beta)
        return self.solve_bicgstab(coef, beta)

    def simple_step(self) -> None:
        """执行原 SIMPLE 的压力修正步骤。"""
        flow = self.flow
        inv_blocks = self.inverse_momentum_blocks()
        ur_e, ur_w, ut_n, ut_s, uz_t, uz_b = flow.rhie_chow(
            flow.UR_tilde,
            flow.UT_tilde,
            flow.UZ_tilde,
            flow.P,
            flow.A11,
            flow.A22,
            flow.A33,
        )
        fe, fw, fn, fs, ft, fb = flow._face_fluxes(ur_e, ur_w, ut_n, ut_s, uz_t, uz_b)
        phi_e, phi_w, phi_n, phi_s, phi_t, phi_b = flow._face_opening()
        beta = -(fe * phi_e + fw * phi_w + fn * phi_n + fs * phi_s + ft * phi_t + fb * phi_b)
        coef = self.coefficients(inv_blocks, beta)
        p_prime = self.solve_system(coef, beta)
        flow.P_prime = p_prime

        d_ur, d_ut, d_uz = self.velocity_correction(p_prime, inv_blocks)
        flow.UR = flow.UR + flow.u_relax * (flow.UR_tilde + d_ur - flow.UR)
        flow.UT = flow.UT + flow.u_relax * (flow.UT_tilde + d_ut - flow.UT)
        flow.UZ = flow.UZ + flow.u_relax * (flow.UZ_tilde + d_uz - flow.UZ)
        flow.P = flow.P + flow.p_relax * p_prime
        flow._apply_boundary()

    def project(
        self,
        beta,
        relax: float,
        velocity_update_limit: float | None = None,
        pressure_update_limit: float | None = None,
    ) -> torch.Tensor:
        """COUPLE 伪瞬态中的压力投影，带速度/压力更新限幅。"""
        flow = self.flow
        inv_blocks = self.inverse_momentum_blocks()
        coef = self.coefficients(inv_blocks, beta)
        p_prime = self.solve_system(coef, beta)
        d_ur, d_ut, d_uz = self.velocity_correction(p_prime, inv_blocks)
        d_ur = relax * d_ur
        d_ut = relax * d_ut
        d_uz = relax * d_uz
        d_p = relax * p_prime
        scale = min(
            flow._velocity_update_scale(d_ur, d_ut, d_uz, velocity_update_limit),
            flow._pressure_update_scale(d_p, pressure_update_limit),
        )
        if scale <= 0.0:
            flow.P_prime = torch.zeros_like(p_prime)
            return flow.P_prime
        flow.P_prime = scale * p_prime
        flow.UR = flow.UR + scale * d_ur
        flow.UT = flow.UT + scale * d_ut
        flow.UZ = flow.UZ + scale * d_uz
        flow.P = flow.P + scale * d_p
        flow._apply_boundary()
        return flow.P_prime
