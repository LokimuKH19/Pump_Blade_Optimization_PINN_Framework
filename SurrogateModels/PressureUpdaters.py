# PressureUpdaters.py
import torch


def _neighbor(x, direction):
    return torch.roll(
        x,
        shifts={"E": -1, "W": 1, "N": -1, "S": 1}[direction],
        dims={"E": 0, "W": 0, "N": 1, "S": 1}[direction],
    )
# 压力求解器合集
# 用于求解位移线性系统 acPc = sum afPf + sum affPff + beta


# 设法兼容一下3D和2D
class Jacobi:
    def __init__(self, coef, beta, device, max_inner=40000, tol=1e-8, report_interval=500):
        """
            coef: dict, including ac, af, aff
            beta: RHS
        """
        self.coef = coef
        self.beta = beta
        self.device = device
        self.max_inner = max_inner
        self.tol = tol
        self.report_interval = report_interval

    # 测试用的雅各比迭代(2D用于验证程序正确性), 参数为P_stencil模板, 2d形态，每轮迭代作为预条件器使用
    def solve2d(self, P_prime, P_prime_old):
        alpha_C = self.coef["C"]
        alpha_E = self.coef["E"]
        alpha_W = self.coef["W"]
        alpha_N = self.coef["N"]
        alpha_S = self.coef["S"]
        alpha_NE = self.coef["NE"]
        alpha_NW = self.coef["NW"]
        alpha_SE = self.coef["SE"]
        alpha_SW = self.coef["SW"]
        beta = self.beta
        # 约束边界
        alpha_W[0, :], alpha_NW[0, :], alpha_SW[0, :] = 0, 0, 0
        alpha_E[-1, :], alpha_NE[-1, :], alpha_SE[-1, :] = 0, 0, 0

        for inner_iter in range(self.max_inner):
            # 获取邻居（周期性已在 neighbor 中处理）
            P_E = _neighbor(P_prime, "E")
            P_W = _neighbor(P_prime, "W")
            P_N = _neighbor(P_prime, "N")
            P_S = _neighbor(P_prime, "S")
            P_NE = _neighbor(_neighbor(P_prime_old, "N"), "E")
            P_NW = _neighbor(_neighbor(P_prime_old, "N"), "W")
            P_SE = _neighbor(_neighbor(P_prime_old, "S"), "E")
            P_SW = _neighbor(_neighbor(P_prime_old, "S"), "W")

            # 径向边界镜像修正（与 momentum 一致）
            P_E[-1, :] = P_prime[-1, :]  # 外边界 E 面用外壁值
            P_W[0, :] = P_prime[0, :]  # 内边界 W 面用内壁值
            # 角点也需相应修正（因涉及 NE, SE 等）
            P_NE[-1, :] = P_N[-1, :]
            P_SE[-1, :] = P_S[-1, :]
            P_NW[0, :] = P_N[0, :]
            P_SW[0, :] = P_S[0, :]

            rhs = (alpha_E * P_E + alpha_W * P_W + alpha_N * P_N + alpha_S * P_S +
                   alpha_NE * P_NE + alpha_NW * P_NW + alpha_SE * P_SE + alpha_SW * P_SW +
                   beta)
            P_prime_new = rhs / (alpha_C + 1e-12)
            omega = 0.3
            P_prime = (1 - omega) * P_prime + omega * P_prime_new

            # ===== 内迭代收敛监控 =====
            residual = rhs-P_prime*alpha_C
            res_inner = torch.max(torch.abs(residual)).item()
            res = torch.mean(torch.abs(P_prime - P_prime_new))

            # 验证压力方程
            if (inner_iter+1) % self.report_interval == 0:
                print(f"  Jacobi inner_iter={inner_iter+1}, res={res:.3e}")

            # 内迭代提前停止说明
            if res < self.tol:
                print(f"  inner converged at {inner_iter+1}, res={res_inner:.3e}")
                break
            # 固定一个压力点
            P_prime[0, 0] = 0.0

        print("Maximum P_prime: ", torch.max(torch.abs(P_prime)).cpu().numpy(),
              ", Avg P_prime: ", torch.mean(torch.abs(P_prime)).cpu().numpy())
        return P_prime


# 工程中用的BiCGStab
class BiCGStab:
    def __init__(self, coef, beta, device="cuda", max_inner=40000, tol=1e-8, report_interval=500):
        self.coef = coef
        self.beta = beta
        self.device = device
        self.max_inner = max_inner
        self.tol = tol
        self.report_interval = report_interval

    def apply_A_2D(self, P):
        """
        计算 A @ P
        """
        alpha_C = self.coef["C"]
        alpha_E = self.coef["E"]
        alpha_W = self.coef["W"]
        alpha_N = self.coef["N"]
        alpha_S = self.coef["S"]
        alpha_NE = self.coef["NE"]
        alpha_NW = self.coef["NW"]
        alpha_SE = self.coef["SE"]
        alpha_SW = self.coef["SW"]

        # neighbor
        def nb(x, d):
            return torch.roll(x, shifts={"E": -1, "W": 1, "N": -1, "S": 1}[d],
                              dims={"E": 0, "W": 0, "N": 1, "S": 1}[d])

        P_E = nb(P, "E")
        P_W = nb(P, "W")
        P_N = nb(P, "N")
        P_S = nb(P, "S")

        P_NE = nb(nb(P, "N"), "E")
        P_NW = nb(nb(P, "N"), "W")
        P_SE = nb(nb(P, "S"), "E")
        P_SW = nb(nb(P, "S"), "W")

        # 径向边界镜像 (零梯度边界条件)
        P_E[-1, :] = P[-1, :]
        P_W[0, :] = P[0, :]

        P_NE[-1, :] = P_N[-1, :]
        P_SE[-1, :] = P_S[-1, :]
        P_NW[0, :] = P_N[0, :]
        P_SW[0, :] = P_S[0, :]

        AP = (
                alpha_C * P
                - (alpha_E * P_E + alpha_W * P_W +
                   alpha_N * P_N + alpha_S * P_S +
                   alpha_NE * P_NE + alpha_NW * P_NW +
                   alpha_SE * P_SE + alpha_SW * P_SW)
        )

        # 【核心修复1】：在矩阵映射层面锁定 [0,0] 点，打破奇异性
        AP[0, 0] = P[0, 0]

        return AP

    def solve2d(self, x0):
        """
        BiCGStab 主求解器
        """
        # 在SIMPLE算法中，压力修正量 P' 的初值永远应该给 0，不要继承上一步的 P'
        x = torch.zeros_like(x0)

        b = self.beta.clone()
        # 对应 AP[0,0] = P[0,0]，将源项的这个点设为 0
        b[0, 0] = 0.0

        r = b - self.apply_A_2D(x)
        r_hat = r.clone()

        rho_old = torch.tensor(1.0, device=self.device)
        alpha = torch.tensor(1.0, device=self.device)
        omega = torch.tensor(1.0, device=self.device)

        v = torch.zeros_like(x)
        p = torch.zeros_like(x)
        res = torch.norm(r)
        for k in range(self.max_inner):
            rho_new = torch.sum(r_hat * r)
            if (k + 1) % self.report_interval == 0:
                print(f"  BiCG iter={k + 1}, res={res.item():.3e}")
            # 安全检查：允许负数，但如果是真正的 0 则中断避免除零错误
            if torch.abs(rho_new) < 1e-10 or torch.abs(rho_old) < 1e-10 or torch.abs(omega) < 1e-10:
                break
            beta_k = (rho_new / rho_old) * (alpha / omega)
            p = r + beta_k * (p - omega * v)
            v = self.apply_A_2D(p)
            den_alpha = torch.sum(r_hat * v)
            if torch.abs(den_alpha) < 1e-10:
                break
            alpha = rho_new / den_alpha
            s = r - alpha * v
            if torch.norm(s) < self.tol:
                x = x + alpha * p
                break
            t = self.apply_A_2D(s)
            tt = torch.sum(t * t)
            if torch.abs(tt) < 1e-10:
                break
            omega = torch.sum(t * s) / tt

            x = x + alpha * p + omega * s
            r = s - omega * t

            rho_old = rho_new

            res = torch.norm(r)

            if res < self.tol:
                print(f"BiCG converged at iter={k + 1}, res={res.item():.3e}")
                break

        print(
            f"Maximum P_prime: {torch.max(torch.abs(x)).item():.6f}, Avg P_prime: {torch.mean(torch.abs(x)).item():.6f}")
        return x


class GMG:
    def __init__(self, coef, device="cuda", levels=3, pre_smooth=3, post_smooth=3):
        """
        coef: dict with keys C, E, W, N, S
        """
        self.device = device
        self.levels = levels
        self.pre = pre_smooth
        self.post = post_smooth

        self.hierarchy = self.build_hierarchy(coef)

    # ===============================
    # 构建多层网格
    # ===============================
    def build_hierarchy(self, coef):
        levels = []
        current = coef

        for _ in range(self.levels):
            levels.append(current)

            # 最粗网格停止
            if current["C"].shape[0] <= 4:
                break

            current = self.coarsen(current)

        return levels

    # ===============================
    # 粗化（2x降采样）
    # ===============================
    def coarsen(self, coef):
        def downsample(x):
            return x[::2, ::2]

        return {
            "C": downsample(coef["C"]),
            "E": downsample(coef["E"]),
            "W": downsample(coef["W"]),
            "N": downsample(coef["N"]),
            "S": downsample(coef["S"]),
        }

    # ===============================
    # 邻居（周期）
    # ===============================
    def nb(self, x, d):
        return torch.roll(
            x,
            shifts={"E": -1, "W": 1, "N": -1, "S": 1}[d],
            dims={"E": 0, "W": 0, "N": 1, "S": 1}[d]
        )

    # ===============================
    # A @ x
    # ===============================
    def apply_A(self, coef, x):
        C = coef["C"]
        E = coef["E"]
        W = coef["W"]
        N = coef["N"]
        S = coef["S"]

        xE = self.nb(x, "E")
        xW = self.nb(x, "W")
        xN = self.nb(x, "N")
        xS = self.nb(x, "S")

        # 径向镜像
        xE[-1, :] = x[-1, :]
        xW[0, :] = x[0, :]

        Ax = C * x - (E * xE + W * xW + N * xN + S * xS)

        # 锁点
        Ax[0, 0] = x[0, 0]

        return Ax

    # ===============================
    # Jacobi smoother
    # ===============================
    def smooth(self, coef, b, x):
        C = coef["C"]
        E = coef["E"]
        W = coef["W"]
        N = coef["N"]
        S = coef["S"]

        for _ in range(self.pre):
            xE = self.nb(x, "E")
            xW = self.nb(x, "W")
            xN = self.nb(x, "N")
            xS = self.nb(x, "S")

            xE[-1, :] = x[-1, :]
            xW[0, :] = x[0, :]

            rhs = E * xE + W * xW + N * xN + S * xS + b
            x_new = rhs / (C + 1e-12)

            # 松弛
            x = 0.7 * x_new + 0.3 * x

            x[0, 0] = 0.0

        return x

    # ===============================
    # Restriction（下采样）
    # ===============================
    def restrict(self, r):
        return r[::2, ::2]

    # ===============================
    # Prolongation（插值）
    # ===============================
    def prolong(self, ec, shape):
        ef = torch.zeros(shape, device=self.device)

        # 1️⃣ 粗点直接赋值
        ef[::2, ::2] = ec

        # 2️⃣ x方向插值（行方向）
        ef[1:-1:2, ::2] = 0.5 * (ec[:-1, :] + ec[1:, :])

        # 边界（最后一行复制）
        ef[-1, ::2] = ec[-1, :]

        # 3️⃣ y方向插值（列方向）
        ef[:, 1:-1:2] = 0.5 * (ef[:, :-2:2] + ef[:, 2::2])

        # 边界（最后一列复制）
        ef[:, -1] = ef[:, -2]

        return ef

    # ===============================
    # V-cycle
    # ===============================
    def v_cycle(self, level, b, x):

        coef = self.hierarchy[level]

        # 最粗网格：直接迭代
        if level == len(self.hierarchy) - 1:
            for _ in range(20):
                x = self.smooth(coef, b, x)
            return x

        # 1️⃣ 预平滑
        x = self.smooth(coef, b, x)

        # 2️⃣ 残差
        r = b - self.apply_A(coef, x)

        # 3️⃣ restrict
        r_c = self.restrict(r)

        # 4️⃣ coarse solve
        ec = torch.zeros_like(r_c)
        ec = self.v_cycle(level + 1, r_c, ec)

        # 5️⃣ prolong
        x = x + self.prolong(ec, x.shape)

        # 6️⃣ 后平滑
        for _ in range(self.post):
            x = self.smooth(coef, b, x)

        return x

    # ===============================
    # 外部接口
    # ===============================
    def solve(self, b, max_iter=20):
        x = torch.zeros_like(b)

        for k in range(max_iter):
            x = self.v_cycle(0, b, x)

            r = b - self.apply_A(self.hierarchy[0], x)
            res = torch.norm(r)

            print(f"  GMG iter={k+1}, res={res.item():.3e}")

            if res < 1e-8:
                break

        return x
