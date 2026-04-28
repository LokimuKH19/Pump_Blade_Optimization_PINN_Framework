# DataGenerator.py
from BladeImport import attach_blade_to_solver
import matplotlib.pyplot as plt
import numpy as np
import torch

import PressureUpdaters


# 最早是个二维程序，现在要按照pdf内容改成三维程序
class BladeCalc:
    """
    三维叶片环形库埃特流动求解器
    使用有限体积法和SIMPLE算法
    坐标为归一化柱坐标 R 和 Theta
    注意周期性边界条件的Theta控制体j的特殊性
    i = 1~N-2
    j = 0~N-1
    k = 1~N-2
    """

    def __init__(self, n=64, rh=2.0, rs=4.0, h=8.0,
                 mu=1.0, rho=1.0, omega=1.0, qv=1.0, n_blade=1,
                 max_iter=5000, tol=1e-6,
                 u_relax=0.5, p_relax=0.3,
                 device="cuda", z0=0.0, blade_params="blade_params.json"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("---Initializing Constants...---")
        self.g = 9.8    # 重力加速度
        self.n = n    # 旋转平面的点数
        self.rh = rh   # 轮毂半径
        self.rs = rs   # 轮缘半径
        self.h = h    # 流道总高度
        self.z0 = z0
        self.mu = mu    # 动力粘度
        self.rho = rho    # 密度
        self.nu = mu / rho    # 运动粘度
        self.omega = omega    # 转子角速度
        self.qv = qv  # 出口体积流量
        self.n_blade = n_blade    # 叶片数，没有叶片的时候为1否则为叶片数
        self.theta0 = 2*np.pi/self.n_blade    # 角度范围
        self.blade_params = blade_params  # 叶片定义

        self.max_iter = max_iter    # 最大迭代轮
        self.tol = tol    # 允差
        self.u_relax = u_relax   # 速度松弛因子
        self.p_relax = p_relax   # 压力松弛因子

        self.delta_r = rs - rh    # 流道宽度
        self.u_omega = rs * self.omega    # 参考旋转平面速度
        self.u_zo = self.qv / np.pi / (self.rs**2-self.rh**2)    # 参考轴向速度
        self.P0 = self.rho * (self.u_omega ** 2 / 2 + self.u_zo ** 2 / 2 + self.g * self.h)    # 参考压力
        self.Re_omega = self.u_omega * self.delta_r / self.nu    # 旋转雷诺数
        self.Eu_omega = self.P0 / (self.rho * self.u_omega ** 2)    # 欧拉数
        self.Gamma = self.delta_r / self.h    # 流道宽高比
        self.Ku = self.u_zo / self.u_omega    # 参考速度比
        self.delta = self.delta_r / self.rs
        self.sgn_omega = 1.0 if self.omega >= 0 else -1.0

        # 生成该求解的东西
        print("---Preparing Variables...---")
        self.P = torch.zeros((n, n, n), device=self.device)
        self.UR = 0.01*torch.randn((n, n, n), device=self.device)
        self.UT = 0.01*torch.randn((n, n, n), device=self.device)
        self.UR_tilde = torch.zeros((n, n, n), device=self.device)    # 存储动量方程专用更新
        self.UT_tilde = torch.zeros((n, n, n), device=self.device)    # 存储动量方程专用更新
        self.Uz_tilde = torch.zeros((n, n, n), device=self.device)
        self.P_prime = torch.zeros((n, n, n), device=self.device)    # 存储压力修正值
        self.UR_prime = torch.zeros((n, n, n), device=self.device)    # 存储动量方程更新系数 # todo 可能要改，不一定用得上
        self.UT_prime = torch.zeros((n, n, n), device=self.device)    # 存储动量方程更新系数

        # 生成系数矩阵
        self.A11 = torch.ones((n, n, n), device=self.device)
        self.A22 = torch.ones((n, n, n), device=self.device)
        self.A12 = torch.ones((n, n, n), device=self.device)
        self.A21 = torch.ones((n, n, n), device=self.device)
        self.A33 = torch.ones((n, n, n), device=self.device)

        # 叶片定义
        self.blade_mask = None
        self.blade_distance = None
        self.blade_distance_z = None
        self.blade_footprint = None
        self.blade_boundary_meta = None
        self.blade_boundary_band_cells = 1.5    # 边界影响区域

        # 启动
        print("---Creating Dimensionless Mesh Automatically...---")
        self._build_grid()
        self._apply_boundary()

        # 几何坐标网格定义
        self.RE = torch.roll(self.R, -1, dims=0)
        self.RW = torch.roll(self.R, 1, dims=0)
        self.RC = self.R
        self.r_hatC = self.R + self.rh / self.delta_r
        self.r_hatE = self.RE + self.rh / self.delta_r
        self.r_hatW = self.RW + self.rh / self.delta_r
        # 相应的界面值
        self.r_hatEf = 0.5 * (self.r_hatC + self.neighbor(self.r_hatC, "E"))
        self.r_hatWf = 0.5 * (self.r_hatC + self.neighbor(self.r_hatC, "W"))
        # 边界处修正：内边界西界面在 R=0 处，r̂ = rh/ΔR
        self.r_hatWf[0, :] = self.rh / self.delta_r
        # 外边界东界面在 R=1 处，r̂ = rs/ΔR
        self.r_hatEf[-1, :] = self.rs / self.delta_r
        self.r_hatW[0, :] = self.rh / self.delta_r
        self.r_hatE[-1, :] = self.rs / self.delta_r
        self.K_theta_C = self.K_theta
        self.K_theta_Ef = 1.0 / (self.r_hatEf * self.theta0)
        self.K_theta_Wf = 1.0 / (self.r_hatWf * self.theta0)

        # 面积（内部点对应的界面）
        self.AE = self.r_hatEf * self.dTheta * self.dZ
        self.AW = self.r_hatWf * self.dTheta * self.dZ
        self.AN = self.dR * self.dZ
        self.AS = self.dR * self.dZ
        self.AT = self.r_hat * self.dTheta * self.dR
        self.AB = self.r_hat * self.dTheta * self.dR
        # 控制体无量纲体积
        self.dV = self.r_hatC * self.dR * self.dTheta * self.dZ

        # 确认边界
        print("---Importing Blades...---")
        self.boundary = self.attach_blade_boundary(self.blade_params, self.blade_boundary_band_cells)

        # 计算蒙版
        print("---Creating the Blade Mask---")
        C = 1    # 日后可以学习
        epsilon = 0.025    # 经验边界层厚度[0.001~0.3]，在算子学习AI版本中可以在CFD中学习
        self.phi_mask = 1 - torch.exp(-C*torch.tensor(self.boundary.signed_distance, device=self.device)**2/epsilon**2)

    # 第一步：生成n*n网格
    def _build_grid(self):
        N = self.n
        self.dR = 1.0 / (N - 1)
        self.dTheta = 1.0 / (N - 1)
        self.dZ = 1.0 / (N - 1)
        R = torch.linspace(0, 1, N, device=self.device)
        Theta = torch.linspace(0, 1, N, device=self.device)
        Z = torch.linspace(0, 1, N, device=self.device)
        RR, TT = torch.meshgrid(R, Theta, indexing='ij')
        self.R = RR
        self.Theta = TT
        self.Z = Z
        self.r_hat = RR + self.rh / self.delta_r
        self.K_theta = 1.0 / (self.r_hat * self.theta0)

    # 传输给，BladeImport.py，这样隔壁就知道归一化的标准是怎样的了
    def set_blade_boundary(self, boundary_data, band_cells=1.5):
        def read_field(name):
            if isinstance(boundary_data, dict):
                return boundary_data[name]
            # 正常来说是个类，不是字典
            return getattr(boundary_data, name)

        self.blade_mask = torch.as_tensor(read_field("mask"), dtype=torch.bool, device=self.device)
        self.blade_distance = torch.as_tensor(
            read_field("signed_distance"),    # 新版欧氏距离
            dtype=torch.float32,
            device=self.device,
        )
        self.blade_distance_z = torch.as_tensor(
            read_field("signed_distance_z"),    # 旧版z距离
            dtype=torch.float32,
            device=self.device,
        )
        self.blade_footprint = torch.as_tensor(
            read_field("footprint_mask"),
            dtype=torch.bool,
            device=self.device,
        )
        self.blade_boundary_meta = read_field("metadata")
        self.blade_boundary_band_cells = float(band_cells)
        self._apply_boundary()

    # 传入blade, 输出标记的boundary
    def attach_blade_boundary(self, blade_params_path="blade_params.json", band_cells=1.5):
        return attach_blade_to_solver(self, blade_params_path, band_cells=band_cells)

    # 辅助整定边界条件
    def _apply_boundary(self):
        self.UR[0, :, :] = 0.0
        self.UR[-1, :, :] = 0.0
        self.UT[0, :, :] = self.rh*self.omega    # 转子角速度
        self.UT[-1, :, :] = 0.0

        if self.blade_mask is None:
            return

        # 边界硬约束
        distance_field = self.blade_distance_z if self.blade_distance_z is not None else self.blade_distance
        # todo 这个distance的约束要重新写一下
        if distance_field is not None:
            band_width = max(self.blade_boundary_band_cells * self.dZ, self.dZ)
            # todo 开始改吧
            wall_weight = torch.clamp(distance_field / band_width, min=0.0, max=1.0)
            self.UR = self.UR * wall_weight
            self.UT = self.UT * wall_weight
            self.UR_tilde = self.UR_tilde * wall_weight
            self.UT_tilde = self.UT_tilde * wall_weight

        self.UR[self.blade_mask] = 0.0
        self.UT[self.blade_mask] = 0.0    # todo 应该是ωr归一化的结果，需要推导一下
        self.UR_tilde[self.blade_mask] = 0.0
        self.UT_tilde[self.blade_mask] = 0.0
        self.P_prime[self.blade_mask] = 0.0

    # 用于找邻居的方法
    @staticmethod
    def neighbor(x, direction):
        dims = {"E": 0, "W": 0, "N": 1, "S": 1}
        forward = {"E": -1, "N": -1, "W": 1, "S": 1}
        return torch.roll(x, forward[direction], dims[direction])

    # 第二步：Rhie-Chow插值
    def rhie_chow(self, UR, UT, P, A11, A12, A21, A22):
        dR = self.dR
        dTheta = self.dTheta
        K_theta = self.K_theta_C
        Eu = self.Eu_omega
        dV = self.dV
        neighbor = self.neighbor

        # ---------- 径向邻居（需要边界修正） ----------
        P_ip = neighbor(P, "E")
        P_im = neighbor(P, "W")
        UR_ip = neighbor(UR, "E")
        UR_im = neighbor(UR, "W")
        A11_ip = neighbor(A11, "E")
        A11_im = neighbor(A11, "W")
        A12_ip = neighbor(A12, "E")
        A12_im = neighbor(A12, "W")
        A21_ip = neighbor(A21, "E")
        A21_im = neighbor(A21, "W")
        A22_ip = neighbor(A22, "E")
        A22_im = neighbor(A22, "W")

        # ---------- 周向邻居（周期性，无需修正） ----------
        P_jp = neighbor(P, "N")
        P_jm = neighbor(P, "S")
        UT_jp = neighbor(UT, "N")
        UT_jm = neighbor(UT, "S")
        A11_jp = neighbor(A11, "N")
        A11_jm = neighbor(A11, "S")
        A12_jp = neighbor(A12, "N")
        A12_jm = neighbor(A12, "S")
        A21_jp = neighbor(A21, "N")
        A21_jm = neighbor(A21, "S")
        A22_jp = neighbor(A22, "N")
        A22_jm = neighbor(A22, "S")

        # ---------- 径向边界镜像修正 ----------
        P_ip[-1, :] = P[-1, :]
        P_im[0, :] = P[0, :]
        UR_ip[-1, :] = UR[-1, :]
        UR_im[0, :] = UR[0, :]
        A11_ip[-1, :] = A11[-1, :]
        A11_im[0, :] = A11[0, :]
        A12_ip[-1, :] = A12[-1, :]
        A12_im[0, :] = A12[0, :]
        A21_ip[-1, :] = A21[-1, :]
        A21_im[0, :] = A21[0, :]
        A22_ip[-1, :] = A22[-1, :]
        A22_im[0, :] = A22[0, :]

        # ======================
        # 2. 梯度，同动量方程，中心FVM，面上FDM
        # ======================
        GR_C = dV * Eu * (P_ip - P_im) / (2 * dR)
        GT_C = dV * Eu * K_theta * (P_jp - P_jm) / (2 * dTheta)

        # 面梯度
        GR_e = dV * Eu * (P_ip - P) / dR
        GR_w = dV * Eu * (P - P_im) / dR

        GT_n = dV * Eu * K_theta * (P_jp - P) / dTheta
        GT_s = dV * Eu * K_theta * (P - P_jm) / dTheta

        # ======================
        # 3. 构造 A^{-1} G P，计算修正速度
        # ======================
        def apply_Ainv(GR, GT, a11, a12, a21, a22):
            det = a11 * a22 - a12 * a21
            det = torch.clamp(det, min=1e-12)
            _UR_ = (a22 * GR - a12 * GT) / det
            _UT_ = (-a21 * GR + a11 * GT) / det
            return _UR_, _UT_

        # 中心
        URc_corr, UTc_corr = apply_Ainv(GR_C, GT_C, A11, A12, A21, A22)

        # 东邻点
        UR_ip_corr, UT_ip_corr = apply_Ainv(
            neighbor(GR_C, "E"), neighbor(GT_C, "E"),
            A11_ip, A12_ip, A21_ip, A22_ip
        )
        # 西邻点
        UR_im_corr, UT_im_corr = apply_Ainv(
            neighbor(GR_C, "W"), neighbor(GT_C, "W"),
            A11_im, A12_im, A21_im, A22_im
        )
        # 北邻点
        UR_jp_corr, UT_jp_corr = apply_Ainv(
            neighbor(GR_C, "N"), neighbor(GT_C, "N"),
            A11_jp, A12_jp, A21_jp, A22_jp
        )
        # 南邻点
        UR_jm_corr, UT_jm_corr = apply_Ainv(
            neighbor(GR_C, "S"), neighbor(GT_C, "S"),
            A11_jm, A12_jm, A21_jm, A22_jm
        )
        # 面系数需要算术平均
        A11_e = 0.5 * (A11 + A11_ip)
        A12_e = 0.5 * (A12 + A12_ip)
        A21_e = 0.5 * (A21 + A21_ip)
        A22_e = 0.5 * (A22 + A22_ip)

        A11_w = 0.5 * (A11 + A11_im)
        A12_w = 0.5 * (A12 + A12_im)
        A21_w = 0.5 * (A21 + A21_im)
        A22_w = 0.5 * (A22 + A22_im)

        A11_n = 0.5 * (A11 + A11_jp)
        A12_n = 0.5 * (A12 + A12_jp)
        A21_n = 0.5 * (A21 + A21_jp)
        A22_n = 0.5 * (A22 + A22_jp)

        A11_s = 0.5 * (A11 + A11_jm)
        A12_s = 0.5 * (A12 + A12_jm)
        A21_s = 0.5 * (A21 + A21_jm)
        A22_s = 0.5 * (A22 + A22_jm)

        # 面（直接用面梯度 + 中心A）
        UR_e_corr, UT_e_corr = apply_Ainv(GR_e, GT_C, A11_e, A12_e, A21_e, A22_e)
        UR_w_corr, UT_w_corr = apply_Ainv(GR_w, GT_C, A11_w, A12_w, A21_w, A22_w)
        UR_n_corr, UT_n_corr = apply_Ainv(GR_C, GT_n, A11_n, A12_n, A21_n, A22_n)
        UR_s_corr, UT_s_corr = apply_Ainv(GR_C, GT_s, A11_s, A12_s, A21_s, A22_s)

        # ======================
        # 4. Rhie–Chow 重构
        # ======================
        UR_e = 0.5 * (UR + UR_ip) - (UR_e_corr - 0.5 * (URc_corr + UR_ip_corr))
        UR_w = 0.5 * (UR + UR_im) - (UR_w_corr - 0.5 * (URc_corr + UR_im_corr))

        UT_n = 0.5 * (UT + UT_jp) - (UT_n_corr - 0.5 * (UTc_corr + UT_jp_corr))
        UT_s = 0.5 * (UT + UT_jm) - (UT_s_corr - 0.5 * (UTc_corr + UT_jm_corr))

        return UR_e, UR_w, UT_n, UT_s

    def momentum(self):
        UR = self.UR
        UT = self.UT

        P = self.P
        neighbor = self.neighbor

        # 周期性邻居（在完整网格上操作）
        UR_jp = neighbor(UR, "N")
        UR_jm = neighbor(UR, "S")
        UT_jp = neighbor(UT, "N")
        UT_jm = neighbor(UT, "S")
        P_jp = neighbor(P, "N")
        P_jm = neighbor(P, "S")

        # 径向邻居（修改为能够兼容边界的形式，直接用roll代替邻居，后面再做边界修正）
        UR_ip = neighbor(UR, "E")
        UR_im = neighbor(UR, "W")
        UT_ip = neighbor(UT, "E")
        UT_im = neighbor(UT, "W")
        P_ip = neighbor(P, "E")
        P_im = neighbor(P, "W")

        # 内边界（i=0，布置ghost point）
        UR_im[0, :] = UR[0, :]
        UT_im[0, :] = UT[0, :]
        P_im[0, :] = P[0, :]
        # 外边界（i=N-1，布置ghost point）
        UR_ip[-1, :] = UR[-1, :]
        UT_ip[-1, :] = UT[-1, :]
        P_ip[-1, :] = P[-1, :]

        # 几何量（全场）
        r_hatC = self.r_hatC
        K_theta_C = self.K_theta

        # 面积（内部点对应的界面）
        AE = self.AE
        AW = self.AW
        AN = self.AN
        AS = self.AS
        # 控制体无量纲体积
        dV = self.dV

        # 无量纲数
        Re = self.Re_omega
        Eu = self.Eu_omega

        # Rhie‑Chow 界面速度（均为 (N, N)）
        uRe, uRw, uTn, uTs = self.rhie_chow(
            UR, UT, P,
            self.A11, self.A12, self.A21, self.A22
        )

        # 通量（已带方向符号，注意 Fn, Fs 已乘 r_hatC）
        Fe = uRe * AE
        Fw = -uRw * AW
        Fn = K_theta_C * uTn * AN * r_hatC
        Fs = -K_theta_C * uTs * AS * r_hatC

        # 扩散系数（已按公式 2.78，周向已包含 r_hatC 和 K_theta^2）
        De = (1.0 / Re) * AE / self.dR
        Dw = (1.0 / Re) * AW / self.dR
        Dn = (1.0 / Re) * AN * r_hatC * (K_theta_C ** 2) / self.dTheta
        Ds = (1.0 / Re) * AS * r_hatC * (K_theta_C ** 2) / self.dTheta

        # 邻点系数 a_F = a_{F,diff} + a_{F,conv} （式 2.74 与 2.77）
        # a_{F,conv} = -min(0, F_F) = clamp(-F_F, min=0) 考虑可读性，此处按文档严格写为 -clamp(F_F, max=0)
        aE = De - torch.clamp(Fe, max=0.0)
        aW = Dw - torch.clamp(Fw, max=0.0)
        aN = Dn - torch.clamp(Fn, max=0.0)
        aS = Ds - torch.clamp(Fs, max=0.0)

        # 对流对中心的贡献 a_{C,conv} = sum max(F_F, 0)
        aP_conv = (torch.clamp(Fe, min=0.0) + torch.clamp(Fw, min=0.0) +
                   torch.clamp(Fn, min=0.0) + torch.clamp(Fs, min=0.0))

        # 扩散对中心的贡献 a_{C,diff} = sum D_F
        aP_diff = De + Dw + Dn + Ds

        # 基础中心系数（不含曲率修正和耦合项）
        aP_base = aP_conv + aP_diff + 1e-12

        # ================== 按照文档 (2.80) 和 (2.82) 构建系数 ==================
        # 内部点当前迭代步速度值（上一迭代步的值）
        UR_C = UR  # 尺寸 (N, N)
        UT_C = UT

        # ---- 径向动量方程系数 a11, a12, b1 ----
        # a11 = a_C,diff + a_C,conv + ΔV/(Re * r_hat^2)
        a11 = aP_base + dV / (Re * (r_hatC ** 2))

        # a12 = - (2 * UT^* / r_hat) * ΔV
        a12 = - (2.0 * UT_C / r_hatC) * dV

        # 源项 b1 = sum_F (a_F * U_R,F) - ΔV * [ Eu * ((r_hat P)_E - (r_hat P)_W)/(2 r_hat ΔR) - (UT^*^2)/r_hat + (
        # K_theta/(Re r_hat)) * (U_Θ,N - U_Θ,S)/ΔΘ ] 注意：右端项中已包含邻点贡献，将其单独写出
        bf1 = aE * UR_ip + aW * UR_im + aN * UR_jp + aS * UR_jm

        # 压力正常有限差分
        pressure_R = Eu * (P_ip - P_im) / (2.0 * self.dR)

        # 曲率显式项: (UT^*^2) / r_hatC
        curve_exp_R = (UT_C ** 2) / r_hatC

        # 交叉导数项: (K_theta/(Re * r_hatC)) * (UT_N - UT_S) / ΔΘ
        UT_N = UT_jp
        UT_S = UT_jm
        cross_deriv_R = (K_theta_C / (Re * r_hatC)) * (UT_N - UT_S) / self.dTheta

        # 源项
        bs1 = - dV * (pressure_R + curve_exp_R + cross_deriv_R)
        # 组合 b1
        b1 = bf1 + bs1

        # ---- 周向动量方程系数 a21, a22, b2 ----
        # a21 = (UT^* / r_hat) * ΔV
        a21 = (UT_C / r_hatC) * dV

        # a22 = aP_base + ΔV/(Re * r_hat^2) + (UR^* / r_hat) * ΔV
        a22 = aP_base + dV / (Re * (r_hatC ** 2)) + (UR_C / r_hatC) * dV

        # 源项 b2 = sum_F (a_F * U_Θ,F) - ΔV * [ Eu * K_theta * (P_N - P_S)/(2 ΔΘ) - (UT^* UR^*)/r_hat - (K_theta/(Re *
        # r_hat)) * (UR_N - UR_S)/ΔΘ ]
        bf2 = aE * UT_ip + aW * UT_im + aN * UT_jp + aS * UT_jm

        # 压力梯度项（周向）
        P_N = P_jp
        P_S = P_jm
        pressure_T = Eu * K_theta_C * (P_N - P_S) / (2.0 * self.dTheta)

        # 耦合项显式部分: (UT^* UR^*) / r_hat
        couple_exp_T = (UT_C * UR_C) / r_hatC

        # 交叉导数项: (K_theta/(Re * r_hat)) * (UR_N - UR_S) / ΔΘ
        UR_N = UR_jp
        UR_S = UR_jm
        cross_deriv_T = (K_theta_C / (Re * r_hatC)) * (UR_N - UR_S) / self.dTheta
        # 源项
        bs2 = - dV * (pressure_T - couple_exp_T - cross_deriv_T)
        # 组合系数
        b2 = bf2 + bs2

        # ================== 联立求解 2x2 线性系统 ==================
        # 方程组：
        # a11 * UR_new + a12 * UT_new = b1
        # a21 * UR_new + a22 * UT_new = b2
        # 求解行列式
        det = a11 * a22 - a12 * a21
        # 防止除零
        det = torch.clamp(det, min=1e-12)
        # 大学第一年学的线性代数be like：这个就是SIMPLE格式中动量方程得到的U_star，不过在文档里写的是\tilde{U}
        self.UR_tilde = (b1 * a22 - a12 * b2) / det
        self.UT_tilde = (a11 * b2 - a21 * b1) / det

        # 保存各系数供压力修正使用以及Rhie-Chow插值
        self.A11, self.A12, self.A21, self.A22 = a11, a12, a21, a22

    def pressure(self):
        # ---------- 引用常用几何量与参数 ----------
        r_hatC = self.r_hatC
        K_theta = self.K_theta_C  # 中心 K_theta
        # K_theta要用界面值
        K_theta_Ef = self.K_theta_Ef
        K_theta_Wf = self.K_theta_Wf
        # 周向界面的 K_theta 与中心相同（因为同一径向位置），但为统一仍用中心值
        K_theta_N = K_theta
        K_theta_S = K_theta

        AE = self.AE
        AW = self.AW
        AN = self.AN
        AS = self.AS
        dV = self.dV
        dR = self.dR
        dTheta = self.dTheta
        Eu = self.Eu_omega
        neighbor = self.neighbor

        # 动量系数（上一迭代步保存）
        A11 = self.A11
        A12 = self.A12
        A21 = self.A21
        A22 = self.A22

        # 计算行列式及逆矩阵元素 (式 2.95)
        det2 = A11 * A22 - A12 * A21 + 1e-12
        a11_r = A22 / det2
        a12_r = -A21 / det2
        a21_r = -A12 / det2
        a22_r = A11 / det2

        # ---------- 1. 左侧：预测通量求和 β = -Σ F̃ ----------
        utRe, utRw, utTn, utTs = self.rhie_chow(
            self.UR_tilde, self.UT_tilde, self.P,
            A11, A12, A21, A22
        )
        Fe_tilde = utRe * AE
        Fw_tilde = -utRw * AW
        Fn_tilde = K_theta * utTn * AN * r_hatC
        Fs_tilde = -K_theta * utTs * AS * r_hatC
        beta = -(Fe_tilde + Fw_tilde + Fn_tilde + Fs_tilde)

        # ---------- 2. 计算压力修正方程系数 (式 2.102-2.105) ----------
        Cv = dV * Eu

        D_RE = Cv * AE / dR
        D_RW = Cv * AW / dR
        D_TN = Cv * r_hatC * AN * (K_theta_N ** 2) / dTheta
        D_TS = Cv * r_hatC * AS * (K_theta_S ** 2) / dTheta

        # 交叉系数，注意使用界面上的 K_theta
        X_12E = Cv * AE * K_theta_Ef / (4.0 * dTheta)
        X_12W = Cv * AW * K_theta_Wf / (4.0 * dTheta)
        X_21N = Cv * r_hatC * AN * K_theta_N / (4.0 * dR)
        X_21S = Cv * r_hatC * AS * K_theta_S / (4.0 * dR)

        # 界面逆矩阵元素（算术平均）
        def face_avg_E(f): return 0.5 * (f + neighbor(f, "E"))
        def face_avg_W(f): return 0.5 * (f + neighbor(f, "W"))
        def face_avg_N(f): return 0.5 * (f + neighbor(f, "N"))
        def face_avg_S(f): return 0.5 * (f + neighbor(f, "S"))

        a11_E = face_avg_E(a11_r)
        a12_E = face_avg_E(a12_r)
        a11_W = face_avg_W(a11_r)
        a12_W = face_avg_W(a12_r)
        a21_N = face_avg_N(a21_r)
        a22_N = face_avg_N(a22_r)
        a21_S = face_avg_S(a21_r)
        a22_S = face_avg_S(a22_r)

        alpha_E = D_RE * a11_E + X_21N * a21_N - X_21S * a21_S
        alpha_W = D_RW * a11_W - X_21N * a21_N + X_21S * a21_S
        alpha_N = X_12E * a12_E - X_12W * a12_W + D_TN * a22_N
        alpha_S = -X_12E * a12_E + X_12W * a12_W + D_TS * a22_S

        alpha_NE = X_12E * a12_E + X_21N * a21_N
        alpha_NW = -X_12W * a12_W - X_21N * a21_N
        alpha_SE = -X_12E * a12_E - X_21S * a21_S
        alpha_SW = X_12W * a12_W + X_21S * a21_S

        # 中心系数
        alpha_C = alpha_E + alpha_W + alpha_N + alpha_S

        # ---------- 3. 迭代求解 P' (带径向边界镜像) ----------
        P_prime = self.P_prime.clone()
        _alpha_ = {"C": alpha_C, "E": alpha_E, "W": alpha_W, "N": alpha_N, "S": alpha_S,
                   "NE": alpha_NE, "NW": alpha_NW, "SE": alpha_SE, "SW": alpha_SW}
        # todo 在这里选择替换压力求解器
        # Jacobi预条件
        P_PRIME_SOLVER = PressureUpdaters.Jacobi(
            _alpha_, beta, self.device,
            max_inner=100,
            tol=1e-6,
            report_interval=100,
        )    # 初始化压力求解器
        P_prime = P_PRIME_SOLVER.solve2d(P_prime, self.P_prime)
        '''
        # BiCG正式
        P_PRIME_SOLVER = PressureUpdaters.BiCGStab(
            _alpha_, beta, self.device,
            max_inner=max_inner,
            tol=1e-10,
            report_interval=1,
        )    # 初始化压力求解器
        P_prime = P_PRIME_SOLVER.solve2d(P_prime)
        '''
        # 只取5点
        coef_gmg = {
            "C": alpha_C,
            "E": alpha_E,
            "W": alpha_W,
            "N": alpha_N,
            "S": alpha_S
        }
        GMG_SOLVER = PressureUpdaters.GMG(coef_gmg, self.device, levels=4)
        P_prime = GMG_SOLVER.solve(beta)

        # ---------- 4. 速度修正 U' = -A^{-1} G P' (式 2.86) ----------
        self.P_prime = P_prime
        P_ip = neighbor(P_prime, "E")
        P_im = neighbor(P_prime, "W")
        P_jp = neighbor(P_prime, "N")
        P_jm = neighbor(P_prime, "S")
        # 边界镜像
        P_ip[-1, :] = P_prime[-1, :]
        P_im[0, :] = P_prime[0, :]

        GR_prime = Cv * (P_ip - P_im) / (2.0 * dR)
        GT_prime = Cv * K_theta * (P_jp - P_jm) / (2.0 * dTheta)

        UR_prime = -(a11_r * GR_prime + a12_r * GT_prime)
        UT_prime = -(a21_r * GR_prime + a22_r * GT_prime)

        # 监控连续性
        mass_res = torch.max(torch.abs(Fe_tilde + Fw_tilde + Fn_tilde + Fs_tilde)).item()
        print(f"mass = {mass_res: .3e}")

        # ---------- 5. 更新速度与压力 ----------
        self.UR = self.UR_tilde + self.u_relax * UR_prime
        self.UT = self.UT_tilde + self.u_relax * UT_prime
        self.P = self.P + self.p_relax * P_prime

    def solve(self):
        for it in range(self.max_iter):

            UR_old = self.UR.clone()
            UT_old = self.UT.clone()

            self.momentum()
            self._apply_boundary()
            res = torch.max(torch.abs(self.UR - UR_old)) + torch.max(torch.abs(self.UT - UT_old))

            print(it, res.item())

            if res < self.tol:
                print("converged", it)
                break

    def post(self):
        N = self.n

        # ===== 构造计算空间网格 =====
        R = np.linspace(0, 1, N)
        Theta = np.linspace(0, 1, N)

        RR, TT = np.meshgrid(R, Theta, indexing='ij')

        # ===== 映射到物理空间 =====
        r = self.rh + RR * (self.rs - self.rh)
        theta = TT * self.theta0

        X = r * np.cos(theta)
        Y = r * np.sin(theta)

        k_mid = self.n // 2

        # ===== 数据 =====
        UT = self.UT[:, :, k_mid].cpu().numpy()
        UR = self.UR[:, :, k_mid].cpu().numpy()
        P = self.P[:, :, k_mid].cpu().numpy()

        # 映射回标准空间
        ut = UT * self.u_omega
        ur = UR * self.u_omega
        p = P * self.P0

        # ===== 1️⃣ 周向速度 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, ut, shading='auto', cmap='cividis')
        plt.colorbar(label="ut/m s-1")
        plt.title("Tangential Velocity (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # ===== 2️⃣ 径向速度 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, ur, shading='auto', cmap='cividis')
        plt.colorbar(label="ur/m s-1")
        plt.title("Radial Velocity (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

        # ===== 3️⃣ 压力 =====
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, p, shading='auto', cmap='cividis')
        plt.colorbar(label="p/Pa")
        plt.title("Pressure (Physical Space)")
        plt.axis('equal')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    # 用来展示叶片边界识别的效果，改成R向的
    def plot_blade_boundary(self, span=0.5):
        if self.blade_mask is None:
            raise RuntimeError("No blade boundary has been attached yet.")
        r_index = int(self.n * span) if span != 1.0 else -1    # 取span=0.5的地方
        mask_slice = self.blade_mask[r_index, :, :].detach().cpu().numpy().T
        dist_slice = self.blade_distance[r_index, :, :].detach().cpu().numpy().T
        phi_mask = self.phi_mask[r_index, :, :].detach().cpu().numpy().T

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

        img = axes[2].imshow(phi_mask, origin="lower", aspect="auto", cmap="GnBu")
        axes[2].set_title(f"Blade Phi Mask @ i={r_index} (span={span})")
        axes[2].set_xlabel("Theta index j")
        axes[2].set_ylabel("Z index k")
        fig.colorbar(img, ax=axes[2], fraction=0.046, pad=0.04)
        fig.tight_layout()
        plt.show()


# 确定随机数的函数
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(10492)
    N = 256
    solver = BladeCalc(
        n=N,
        rh=0.0605,
        rs=0.08,
        h=0.125,
        mu=1.0,
        rho=1.0,
        omega=1.0,
        n_blade=6,
        max_iter=200,
        tol=1e-5,
        u_relax=0.3,
        p_relax=0.3,
        device="cuda",
        z0=0.0,
        blade_params="../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc/blade_params.json",
    )
    span = 0.5
    Ri = int(N * span) if span != 1.0 else -1
    slice_distance = solver.boundary.signed_distance[Ri, :, :].copy()
    inside_vals = slice_distance[solver.boundary.mask[Ri, :, :]]
    print(f"内部体素距离值: min={inside_vals.min()}, max={inside_vals.max()}, mean={inside_vals.mean()}")
    print(f"Blade boundary attached, masked points = {solver.boundary.metadata['mask_points']}")
    solver.plot_blade_boundary(span=span)
