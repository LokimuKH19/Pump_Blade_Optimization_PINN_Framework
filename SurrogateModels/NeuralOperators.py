# NeuralOperators.py
# 神经算子集合
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_everything(seed):
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(precision=16)
    np.set_printoptions(precision=16)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# DCT/IDCT (DCT-II / DCT-III) 1D and separable 2D implementations, this aims to use fft in chebyshev expansion
# thus we applied the 1st-type Chebyshev polynomials, Tk(x) = cos(k arccos x),
# and it can be writtin in the Discrete Cosine Transform(DCT).
# -------------------------
def dct_1d(x):
    # x: (..., N), real
    N = x.shape[-1]
    v = torch.cat([x, x.flip(-1)], dim=-1)  # (..., 2N)s
    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(N, device=x.device, dtype=x.dtype)
    exp_factor = torch.exp(-1j * math.pi * k / (2 * N))
    X = (V[..., :N] * exp_factor).real
    X[..., 0] *= 0.5
    return X


def idct_1d(X):
    # inverse of dct_1d (DCT-III), X: (..., N)
    N = X.shape[-1]
    c = X.clone()
    c[..., 0] = c[..., 0] * 2.0
    k = torch.arange(N, device=X.device, dtype=X.dtype)
    exp_factor = torch.exp(1j * math.pi * k / (2 * N))
    V = torch.zeros(X.shape[:-1] + (2 * N,), dtype=torch.cfloat, device=X.device)
    V[..., :N] = (c * exp_factor)
    if N > 1:
        V[..., N + 1:] = torch.conj(V[..., 1:N].flip(-1))
    V[..., N] = torch.tensor(0.0 + 0.0j)
    v = torch.fft.ifft(V, dim=-1)
    x = v[..., :N].real
    return x


def dct_2d(x):
    # x: (..., H, W)
    # apply dct along last dim then along -2
    orig_shape = x.shape
    # last dim
    x_resh = x.reshape(-1, orig_shape[-1])
    y = dct_1d(x_resh).reshape(*orig_shape)
    # swap last two and apply again
    y_perm = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    shp = y_perm.shape
    y2 = dct_1d(y_perm.reshape(-1, shp[-1])).reshape(shp)
    return y2.permute(*range(y2.dim() - 2), y2.dim() - 1, y2.dim() - 2)


def idct_2d(X):
    # inverse 2D: apply idct along -2 then -1 (reverse order)
    X_perm = X.permute(*range(X.dim() - 2), X.dim() - 1, X.dim() - 2)
    shp = X_perm.shape
    y = idct_1d(X_perm.reshape(-1, shp[-1])).reshape(shp)
    y = y.permute(*range(y.dim() - 2), y.dim() - 1, y.dim() - 2)
    z = idct_1d(y.reshape(-1, y.shape[-1])).reshape(y.shape)
    return z


def _transform_along_dim(x, dim, transform):
    x_perm = torch.movedim(x, dim, -1)
    shape = x_perm.shape
    y = transform(x_perm.reshape(-1, shape[-1])).reshape(shape)
    return torch.movedim(y, -1, dim)


def dct_3d(x):
    # x: (..., D, H, W)
    y = _transform_along_dim(x, -1, dct_1d)
    y = _transform_along_dim(y, -2, dct_1d)
    y = _transform_along_dim(y, -3, dct_1d)
    return y


def idct_3d(X):
    # inverse 3D DCT in reverse order.
    y = _transform_along_dim(X, -3, idct_1d)
    y = _transform_along_dim(y, -2, idct_1d)
    y = _transform_along_dim(y, -1, idct_1d)
    return y


# -------------------------
# Chebyshev / Cosine spectral conv (real coefficients)
# -------------------------
class ChebSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_h, modes_w):
        """
        in_channels, out_channels: channels
        modes_h, modes_w: number of retained modes in each dim (use <= H, W)
        The weight shape: (in_channels, out_channels, modes_h, modes_w) real
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_h = modes_h
        self.m_w = modes_w
        # real coefficients for Chebyshev/DCT space
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes_h, modes_w) * (1.0 / (in_channels * out_channels) ** 0.5))

    def forward(self, x):
        # x: [B, C, H, W] real
        B, C, H, W = x.shape
        # compute DCT2 on each channel
        # reshape to (..., H, W) to operate with dct_2d
        x_dct = dct_2d(x)  # shape [B, C, H, W]
        # crop modes (take top-left modes_h x modes_w)
        x_modes = x_dct[:, :, :self.m_h, :self.m_w]  # [B, C, m_h, m_w]
        # multiply by real weights: einsum over in_channel
        # out_modes[b, o, i, j] = sum_c x_modes[b, c, i, j] * weight[c, o, i, j]
        out_modes = torch.einsum("b c i j, c o i j -> b o i j", x_modes, self.weight)
        # create full spectral tensor with zeros then place modes back
        out_dct = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)
        out_dct[:, :, :self.m_h, :self.m_w] = out_modes
        # inverse DCT2
        out = idct_2d(out_dct)
        return out


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        """
        in_channels, out_channels: number of channels
        modes: modes that reserved, Assume that H, W >= modes!!!!!
        weights are in complex，symmetric can be recovered by conjugate mirror
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # einsum over in_channel
        # input: [B, in, H, W], weights: [in, out, mh, mw]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        x: [B, C, H, W] (实数)
        """
        B, C, H, W = x.shape
        # 2D FFT (use complex)
        x_ft = torch.fft.rfft2(x, norm="forward")  # [B, C, H, W//2+1]

        # Output a frequency tensor
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )

        # Low frequency modes × modes
        mh, mw = self.modes, self.modes
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(x_ft[:, :, :mh, :mw], self.weights)

        # IFFT
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out


class ChebSpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_r, modes_theta, modes_z):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m_r = int(modes_r)
        self.m_theta = int(modes_theta)
        self.m_z = int(modes_z)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, self.m_r, self.m_theta, self.m_z)
            * (1.0 / max(in_channels * out_channels, 1) ** 0.5)
        )

    def forward(self, x):
        # x: [B, C, R, Theta, Z]
        B, _, R, T, Z = x.shape
        mr = min(self.m_r, R)
        mt = min(self.m_theta, T)
        mz = min(self.m_z, Z)
        x_dct = dct_3d(x)
        x_modes = x_dct[:, :, :mr, :mt, :mz]
        out_modes = torch.einsum(
            "b c i j k, c o i j k -> b o i j k",
            x_modes,
            self.weight[:, :, :mr, :mt, :mz],
        )
        out_dct = torch.zeros(B, self.out_channels, R, T, Z, device=x.device, dtype=x.dtype)
        out_dct[:, :, :mr, :mt, :mz] = out_modes
        return idct_3d(out_dct)


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = int(modes)
        scale = 1 / max(in_channels * out_channels, 1)
        shape = (in_channels, out_channels, self.modes, self.modes, self.modes)
        self.weights_pp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_np = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_pn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_nn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        # x: [B, C, R, Theta, Z]
        B, _, R, T, Z = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
        z_freq = Z // 2 + 1
        out_ft = torch.zeros(B, self.out_channels, R, T, z_freq, device=x.device, dtype=torch.cfloat)

        mr = min(self.modes, max(1, R // 2))
        mt = min(self.modes, max(1, T // 2))
        mz = min(self.modes, z_freq)
        out_ft[:, :, :mr, :mt, :mz] = self.compl_mul3d(
            x_ft[:, :, :mr, :mt, :mz],
            self.weights_pp[:, :, :mr, :mt, :mz],
        )
        out_ft[:, :, -mr:, :mt, :mz] = self.compl_mul3d(
            x_ft[:, :, -mr:, :mt, :mz],
            self.weights_np[:, :, :mr, :mt, :mz],
        )
        out_ft[:, :, :mr, -mt:, :mz] = self.compl_mul3d(
            x_ft[:, :, :mr, -mt:, :mz],
            self.weights_pn[:, :, :mr, :mt, :mz],
        )
        out_ft[:, :, -mr:, -mt:, :mz] = self.compl_mul3d(
            x_ft[:, :, -mr:, -mt:, :mz],
            self.weights_nn[:, :, :mr, :mt, :mz],
        )
        return torch.fft.irfftn(out_ft, s=(R, T, Z), dim=(-3, -2, -1), norm="forward")


def mixed_boundary_pad2d(x, pad_h, pad_w):
    # H is treated as periodic, W as non-periodic. This matches theta-z slices
    # after SurrogateModeling applies replicate padding in Z.
    if pad_w > 0:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="replicate")
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="circular")
    return x


# 说白了就是我做了个带通滤波器：
def mixed_boundary_pad3d(x, pad_r, pad_theta, pad_z):
    # x: [B, C, R, Theta, Z]. R/Z are non-periodic, Theta is periodic.
    if pad_r > 0 or pad_z > 0:
        x = F.pad(x, (pad_z, pad_z, 0, 0, pad_r, pad_r), mode="replicate")
    if pad_theta > 0:
        x = F.pad(x, (0, 0, pad_theta, pad_theta, 0, 0), mode="circular")
    return x


class MultiBandSpectralConv2d(nn.Module):
    # FNO's usual low-mode crop cannot represent sharp IBM boundaries well.
    # This layer keeps low modes and a small high-z band, and includes both
    # positive and negative theta rows in the full FFT plane.
    def __init__(self, in_channels, out_channels, low_modes, high_modes=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_modes = int(low_modes)
        self.high_modes = int(max(high_modes, 0))
        scale = 1 / max(in_channels * out_channels, 1)
        # 左上角低频成分
        self.weights_low_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.low_modes, self.low_modes, dtype=torch.cfloat)
        )
        self.weights_low_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, self.low_modes, self.low_modes, dtype=torch.cfloat)
        )
        # high_modes added into the model structure，已经开始笑了
        if self.high_modes > 0:
            self.weights_high_pos = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.low_modes, self.high_modes, dtype=torch.cfloat)
            )
            self.weights_high_neg = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.low_modes, self.high_modes, dtype=torch.cfloat)
            )
            self.weights_high_y_pos = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.high_modes, self.low_modes, dtype=torch.cfloat)
            )
            self.weights_high_y_neg = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.high_modes, self.low_modes, dtype=torch.cfloat)
            )
            self.weights_high_xy_pos = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.high_modes, self.high_modes, dtype=torch.cfloat)
            )
            self.weights_high_xy_neg = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.high_modes, self.high_modes, dtype=torch.cfloat)
            )
        else:
            self.register_parameter("weights_high_pos", None)
            self.register_parameter("weights_high_neg", None)
            self.register_parameter("weights_high_y_pos", None)
            self.register_parameter("weights_high_y_neg", None)
            self.register_parameter("weights_high_xy_pos", None)
            self.register_parameter("weights_high_xy_neg", None)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="forward")
        freq_w = W // 2 + 1
        out_ft = torch.zeros(B, self.out_channels, H, freq_w, device=x.device, dtype=torch.cfloat)

        mh = min(self.low_modes, H)
        mw = min(self.low_modes, freq_w)
        if mh > 0 and mw > 0:
            out_ft[:, :, :mh, :mw] = self.compl_mul2d(
                x_ft[:, :, :mh, :mw],
                self.weights_low_pos[:, :, :mh, :mw],
            )
            out_ft[:, :, -mh:, :mw] = self.compl_mul2d(
                x_ft[:, :, -mh:, :mw],
                self.weights_low_neg[:, :, :mh, :mw],
            )

        x_high_available = max(freq_w - mw, 0)
        hw = min(self.high_modes, x_high_available)
        y_pos_end = H // 2 + 1
        y_neg_start = H // 2 + 1
        y_pos_available = max(y_pos_end - mh, 0)
        y_neg_available = max((H - mh) - y_neg_start, 0)
        hy = min(self.high_modes, y_pos_available, y_neg_available)

        if mh > 0 and hw > 0:
            out_ft[:, :, :mh, -hw:] = out_ft[:, :, :mh, -hw:] + self.compl_mul2d(
                x_ft[:, :, :mh, -hw:],
                self.weights_high_pos[:, :, :mh, :hw],
            )
            out_ft[:, :, -mh:, -hw:] = out_ft[:, :, -mh:, -hw:] + self.compl_mul2d(
                x_ft[:, :, -mh:, -hw:],
                self.weights_high_neg[:, :, :mh, :hw],
            )
        if hy > 0:
            y_pos_start = y_pos_end - hy
            y_neg_end = y_neg_start + hy
            if mw > 0:
                out_ft[:, :, y_pos_start:y_pos_end, :mw] = out_ft[:, :, y_pos_start:y_pos_end, :mw] + self.compl_mul2d(
                    x_ft[:, :, y_pos_start:y_pos_end, :mw],
                    self.weights_high_y_pos[:, :, :hy, :mw],
                )
                out_ft[:, :, y_neg_start:y_neg_end, :mw] = out_ft[:, :, y_neg_start:y_neg_end, :mw] + self.compl_mul2d(
                    x_ft[:, :, y_neg_start:y_neg_end, :mw],
                    self.weights_high_y_neg[:, :, :hy, :mw],
                )
            if hw > 0:
                out_ft[:, :, y_pos_start:y_pos_end, -hw:] = out_ft[:, :, y_pos_start:y_pos_end, -hw:] + self.compl_mul2d(
                    x_ft[:, :, y_pos_start:y_pos_end, -hw:],
                    self.weights_high_xy_pos[:, :, :hy, :hw],
                )
                out_ft[:, :, y_neg_start:y_neg_end, -hw:] = out_ft[:, :, y_neg_start:y_neg_end, -hw:] + self.compl_mul2d(
                    x_ft[:, :, y_neg_start:y_neg_end, -hw:],
                    self.weights_high_xy_neg[:, :, :hy, :hw],
                )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="forward")


class FourierFeatureGrid2d(nn.Module):
    # Fixed coordinate features give the pointwise MLP a direct high-frequency
    # basis instead of forcing the spectral trunk to synthesize all oscillations.
    def __init__(self, bands=(1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("bands", torch.tensor(list(bands), dtype=torch.float32), persistent=False)

    @property
    def extra_channels(self):
        return int(self.bands.numel()) * 4

    def forward(self, x):
        if self.bands.numel() == 0:
            return x
        B, _, H, W = x.shape
        yy = torch.linspace(0.0, 1.0, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1)
        zz = torch.linspace(0.0, 1.0, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W)
        bands = self.bands.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        phase_y = 2.0 * math.pi * bands * yy
        phase_z = 2.0 * math.pi * bands * zz
        y_features = torch.cat([torch.sin(phase_y), torch.cos(phase_y)], dim=1).expand(B, -1, H, W)
        z_features = torch.cat([torch.sin(phase_z), torch.cos(phase_z)], dim=1).expand(B, -1, H, W)
        return torch.cat([x, y_features, z_features], dim=1)


class FourierFeatureGrid3d(nn.Module):
    def __init__(self, bands=(1, 2, 4, 8)):
        super().__init__()
        self.register_buffer("bands", torch.tensor(list(bands), dtype=torch.float32), persistent=False)

    @property
    def extra_channels(self):
        return int(self.bands.numel()) * 6

    def forward(self, x):
        if self.bands.numel() == 0:
            return x
        B, _, R, T, Z = x.shape
        rr = torch.linspace(0.0, 1.0, R, device=x.device, dtype=x.dtype).view(1, 1, R, 1, 1)
        tt = torch.linspace(0.0, 1.0, T, device=x.device, dtype=x.dtype).view(1, 1, 1, T, 1)
        zz = torch.linspace(0.0, 1.0, Z, device=x.device, dtype=x.dtype).view(1, 1, 1, 1, Z)
        bands = self.bands.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1, 1)
        phase_r = 2.0 * math.pi * bands * rr
        phase_t = 2.0 * math.pi * bands * tt
        phase_z = 2.0 * math.pi * bands * zz
        r_features = torch.cat([torch.sin(phase_r), torch.cos(phase_r)], dim=1).expand(B, -1, R, T, Z)
        t_features = torch.cat([torch.sin(phase_t), torch.cos(phase_t)], dim=1).expand(B, -1, R, T, Z)
        z_features = torch.cat([torch.sin(phase_z), torch.cos(phase_z)], dim=1).expand(B, -1, R, T, Z)
        return torch.cat([x, r_features, t_features, z_features], dim=1)


class LocalHighPassBlock2d(nn.Module):
    # A local branch catches edges and short-wavelength content that low-mode
    # spectral layers deliberately remove.
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.pad = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=0)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        smooth = F.avg_pool2d(mixed_boundary_pad2d(x, 1, 1), kernel_size=3, stride=1)
        high = x - smooth
        y = self.depthwise(mixed_boundary_pad2d(high, self.pad, self.pad))
        y = F.gelu(self.pointwise(y))
        return self.mix(y)


class LocalHighPassBlock3d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.pad = kernel_size // 2
        self.depthwise = nn.Conv3d(channels, channels, kernel_size, groups=channels, padding=0)
        self.pointwise = nn.Conv3d(channels, channels, 1)
        self.mix = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        smooth = F.avg_pool3d(mixed_boundary_pad3d(x, 1, 1, 1), kernel_size=3, stride=1)
        high = x - smooth
        y = self.depthwise(mixed_boundary_pad3d(high, self.pad, self.pad, self.pad))
        y = F.gelu(self.pointwise(y))
        return self.mix(y)


class HFCFNOBlock(nn.Module):
    def __init__(
        self,
        channels,
        modes,
        cheb_modes,
        high_modes=4,
        alpha_init=0.5,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        self.low_cfno = CFNOBlock(channels, channels, modes, cheb_modes, alpha_init=alpha_init)
        self.band_spectral = MultiBandSpectralConv2d(channels, channels, modes, high_modes=high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock2d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_cfno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


class HFFNOBlock(nn.Module):
    # FNO trunk plus the same gated high-frequency branches used by HF-CFNO.
    # This isolates whether the Chebyshev path is actually needed.
    def __init__(
        self,
        channels,
        modes,
        high_modes=4,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        self.low_fno = SpectralConv2d(channels, channels, modes)
        self.band_spectral = MultiBandSpectralConv2d(channels, channels, modes, high_modes=high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock2d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_fno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


# -------------------------
# CFNO block: combine Fourier spectral conv and Chebyshev spectral conv per layer
# -------------------------
class CFNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, cheb_modes, alpha_init=0.5):
        # alpha\in[0,1], 0.5 is the default for initialization and self-adaptive fitting
        super().__init__()
        self.fourier = SpectralConv2d(in_channels, out_channels, modes)
        mh, mw = cheb_modes
        self.cheb = ChebSpectralConv2d(in_channels, out_channels, mh, mw)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


class CFNOBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, cheb_modes, alpha_init=0.5):
        super().__init__()
        self.fourier = SpectralConv3d(in_channels, out_channels, modes)
        mr, mt, mz = cheb_modes
        self.cheb = ChebSpectralConv3d(in_channels, out_channels, mr, mt, mz)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.fuse = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        y_f = self.fourier(x)
        y_c = self.cheb(x)
        a = torch.sigmoid(self.alpha)
        y_blend = a * y_f + (1.0 - a) * y_c
        y_cat = torch.cat([y_f, y_c], dim=1)
        y_fused = self.fuse(y_cat)
        return y_blend + y_fused


class HFCFNOBlock3d(nn.Module):
    def __init__(
        self,
        channels,
        modes,
        cheb_modes,
        high_modes=4,
        alpha_init=0.5,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        high_modes = int(max(high_modes, 0))
        self.low_cfno = CFNOBlock3d(channels, channels, modes, cheb_modes, alpha_init=alpha_init)
        self.band_spectral = SpectralConv3d(channels, channels, int(modes) + high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock3d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv3d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_cfno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


class HFFNOBlock3d(nn.Module):
    def __init__(
        self,
        channels,
        modes,
        high_modes=4,
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        high_modes = int(max(high_modes, 0))
        self.low_fno = SpectralConv3d(channels, channels, modes)
        self.band_spectral = SpectralConv3d(channels, channels, int(modes) + high_modes)
        self.use_local_highpass = bool(use_local_highpass)
        self.local_high = LocalHighPassBlock3d(channels) if self.use_local_highpass else None
        self.fuse = nn.Conv3d(channels * 3, channels, 1)
        self.high_gate = nn.Parameter(torch.tensor(float(high_gate_init)))

    def forward(self, x):
        low = self.low_fno(x)
        band = self.band_spectral(x)
        local = self.local_high(x) if self.local_high is not None else torch.zeros_like(band)
        gate = torch.sigmoid(self.high_gate)
        fused = self.fuse(torch.cat([low, band, local], dim=1))
        return low + gate * (band + local) + fused


# -------------------------
# CFNO network (example stack)
# -------------------------
class CFNO2d(nn.Module):
    def __init__(self, modes=12, cheb_modes=(12, 12), width=32, depth=4):
        super().__init__()
        self.width = width
        self.depth = depth
        # input lifting (like your FNO fc0)
        self.fc0 = nn.Linear(2, width)
        # create layer stacks of CFNOBlock with 1x1 conv residuals (similar to FNO architecture)
        self.blocks = nn.ModuleList()
        self.w_convs = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(CFNOBlock(width, width, modes, cheb_modes))
            self.w_convs.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # x: [B, 2, H, W]
        B, C, H, W = x.shape
        # lift
        x = x.permute(0, 2, 3, 1)  # [B, H, W, 2]
        x = self.fc0(x)  # [B, H, W, width]
        x = x.permute(0, 3, 1, 2)  # [B, width, H, W]
        # stack
        for block, w_conv in zip(self.blocks, self.w_convs):
            y = block(x)
            x = y + w_conv(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # [B, H, W, 2]
        x = x.permute(0, 3, 1, 2)  # [B, 2, H, W]
        return x


# -------------------- Exampled Network Construction: FNO, CNO, CFNO --------------------
class FNO2d_small(nn.Module):
    def __init__(self, modes=8, width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(depth)])     # fourier transform
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])    # weights
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):  # x: [B,1,H,W] source f
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,1]
        x = self.fc0(x)  # [B,H,W,width]
        x = x.permute(0, 3, 1, 2)  # [B,width,H,W]
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CNO model: use ChebSpectralConv2d blocks instead of Fourier
class CNO2d_small(nn.Module):
    def __init__(self, cheb_modes=(8, 8), width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList(
            [ChebSpectralConv2d(width, width, cheb_modes[0], cheb_modes[1]) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


# CFNO combining both
class CFNO2d_small(nn.Module):
    def __init__(self, modes=8, cheb_modes=(8, 8), width=16, depth=3, alpha_init=0.5, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([CFNOBlock(width, width, modes, cheb_modes, alpha_init=alpha_init) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class HF_CFNO2d_small(nn.Module):
    # High-frequency enhanced CFNO. It keeps the original CFNO low-mode path,
    # adds fixed Fourier coordinate features, a multi-band FFT branch, and a
    # local high-pass branch.
    def __init__(
        self,
        modes=8,
        cheb_modes=(8, 8),
        high_modes=None,
        width=16,
        depth=3,
        alpha_init=0.5,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid2d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFCFNOBlock(
                    width,
                    modes=modes,
                    cheb_modes=cheb_modes,
                    high_modes=high_modes,
                    alpha_init=alpha_init,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class HF_FNO2d_small(nn.Module):
    # High-frequency enhanced FNO. Same high-frequency/filtering scaffold as
    # HF-CFNO, but the low path is pure Fourier rather than CFNO.
    def __init__(
        self,
        modes=8,
        high_modes=None,
        width=16,
        depth=3,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid2d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFFNOBlock(
                    width,
                    modes=modes,
                    high_modes=high_modes,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FNO3d_small(nn.Module):
    def __init__(self, modes=8, width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList([SpectralConv3d(width, width, modes) for _ in range(depth)])
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        # x: [B, C, R, Theta, Z]
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)


class CNO3d_small(nn.Module):
    def __init__(self, cheb_modes=(8, 8, 8), width=16, depth=3, input_features=1, output_features=1):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList(
            [ChebSpectralConv3d(width, width, cheb_modes[0], cheb_modes[1], cheb_modes[2]) for _ in range(depth)]
        )
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)


class CFNO3d_small(nn.Module):
    def __init__(
        self,
        modes=8,
        cheb_modes=(8, 8, 8),
        width=16,
        depth=3,
        alpha_init=0.5,
        input_features=1,
        output_features=1,
    ):
        super().__init__()
        self.fc0 = nn.Linear(input_features, width)
        self.blocks = nn.ModuleList(
            [CFNOBlock3d(width, width, modes, cheb_modes, alpha_init=alpha_init) for _ in range(depth)]
        )
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, output_features)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = y + w(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)


class HF_CFNO3d_small(nn.Module):
    def __init__(
        self,
        modes=8,
        cheb_modes=(8, 8, 8),
        high_modes=None,
        width=16,
        depth=3,
        alpha_init=0.5,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid3d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFCFNOBlock3d(
                    width,
                    modes=modes,
                    cheb_modes=cheb_modes,
                    high_modes=high_modes,
                    alpha_init=alpha_init,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)


class HF_FNO3d_small(nn.Module):
    def __init__(
        self,
        modes=8,
        high_modes=None,
        width=16,
        depth=3,
        input_features=1,
        output_features=1,
        fourier_feature_bands=(1, 2, 4, 8),
        high_gate_init=-1.0,
        use_local_highpass=True,
    ):
        super().__init__()
        if high_modes is None:
            high_modes = max(2, int(modes) // 2)
        self.feature_grid = FourierFeatureGrid3d(fourier_feature_bands)
        lifted_features = input_features + self.feature_grid.extra_channels
        self.fc0 = nn.Linear(lifted_features, width)
        self.blocks = nn.ModuleList(
            [
                HFFNOBlock3d(
                    width,
                    modes=modes,
                    high_modes=high_modes,
                    high_gate_init=high_gate_init,
                    use_local_highpass=use_local_highpass,
                )
                for _ in range(depth)
            ]
        )
        self.wconvs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        hidden = max(64, width * 2)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, output_features)

    def forward(self, x):
        x = self.feature_grid(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        for blk, w in zip(self.blocks, self.wconvs):
            y = blk(x)
            x = F.gelu(y + w(x))
        x = x.permute(0, 2, 3, 4, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 4, 1, 2, 3)


# -------------------- Poisson dataset generation using Jacobi solver --------------------
def poisson(f, iters=500, tol=1e-6):
    # Solve -Δ u = f on unit square with zero Dirichlet BC using Jacobi for interior points
    H, W = f.shape
    u = torch.zeros_like(f)
    # keep boundary zero
    dx = 1.0 / (H - 1)
    dy = 1.0 / (W - 1)
    dx2 = dx * dx
    dy2 = dy * dy
    denom = 2 * (dx2 + dy2)
    # use Jacobi iteration
    for _ in range(iters):
        u_old = u.clone()
        # NOTE: *** here the sign for f is + (not -) ***
        u[1:-1, 1:-1] = ((u_old[2:, 1:-1] + u_old[:-2, 1:-1]) * dy2 +
                         (u_old[1:-1, 2:] + u_old[1:-1, :-2]) * dx2 +
                         f[1:-1, 1:-1] * dx2 * dy2) / denom
        # boundaries remain zero (Dirichlet)
        if torch.max(torch.abs(u - u_old)) < tol:
            break
    return u


def make_dataset(n_samples=200, H=32, W=32, data_rate=1):
    # create random RHS f with localized sources (sum of gaussians bumps)
    X = []
    Y = []
    i = 0
    for _ in range(n_samples):
        i += 1
        f = torch.zeros(H, W)
        # add a few random gaussian bumps
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.uniform(0.2, 0.8)
            cy = np.random.uniform(0.2, 0.8)
            sx = np.random.uniform(0.03, 0.12)
            sy = np.random.uniform(0.03, 0.12)
            xv = torch.linspace(0, 1, H)
            yv = torch.linspace(0, 1, W)
            Xg, Yg = torch.meshgrid(xv, yv, indexing='ij')
            g = torch.exp(-((Xg - cx) ** 2) / (2 * sx ** 2) - ((Yg - cy) ** 2) / (2 * sy ** 2))
            amp = np.random.uniform(-5, 5)
            f += amp * g
        # solve Poisson
        u = poisson(f, iters=2000, tol=1e-6)
        X.append(f.unsqueeze(0))  # channel dim, add f to X
        Y.append(u.unsqueeze(0))
        print(f"Dataset Construction: {i}/{n_samples}")
    X = torch.stack(X)  # [N,1,H,W]
    Y = torch.stack(Y)  # [N,1,H,W]
    return X, Y


def laplacian(u, dx=1.0, dy=1.0):
    """Compute Laplacian using central differences (2nd order)."""
    return (
        (u[..., :-2, 1:-1] - 2 * u[..., 1:-1, 1:-1] + u[..., 2:, 1:-1]) / dx**2 +
        (u[..., 1:-1, :-2] - 2 * u[..., 1:-1, 1:-1] + u[..., 1:-1, 2:]) / dy**2
    )


# -------------------- Physics-Informed Training utilities (PDE loss) --------------------
def train_model(model, X_train, Y_train, X_val, Y_val, mask, epochs=60, batch_size=8, lr=1e-3, pde_rate=0.001, data_rate=1):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X_train.shape[0]
    logs = {'train': [], 'val': []}
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        train_loss = 0.0
        for i in range(0, n, batch_size):
            if data_rate == 0:
                # Pure Physics, this redundant is for later expansion
                idx = perm[i:i + batch_size]
                xb = X_train[idx].to(device)
                pred = model(xb)
                B, _, H, W = pred.shape
                dx = 1.0 / (H - 1)
                dy = 1.0 / (W - 1)
                u = pred[:, 0:1, :, :]*mask  # [B,1,H,W]
                lap_u = laplacian(u, dx, dy)  # [B,1,H-2,W-2]
                f_interior = xb[:, :, 1:-1, 1:-1]
                loss = loss_fn(-lap_u, f_interior)
            else:
                idx = perm[i:i + batch_size]
                xb = X_train[idx]  # f(x)
                yb = Y_train[idx]
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)  # u_pred, [B,1,H,W]
                B, _, H, W = pred.shape

                dx = 1.0 / (H - 1)
                dy = 1.0 / (W - 1)
                u = pred[:, 0:1, :, :] * mask  # [B,1,H,W]
                lap_u = laplacian(u, dx, dy)  # shape [B,1,H-2,W-2]

                f_interior = xb[:, :, 1:-1, 1:-1]  # assuming xb holds f
                loss_pde = loss_fn(-lap_u, f_interior)

                loss_data = loss_fn(u, yb * mask)
                loss = loss_pde * pde_rate + loss_data * data_rate
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.shape[0]

        train_loss /= n

        # validation (reporting)
        model.eval()
        with torch.no_grad():
            predv = model(X_val)
            mse_val = loss_fn(predv * mask, Y_val * mask).item()

        logs['train'].append(train_loss)
        logs['val'].append(mse_val)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{epochs}  Train_loss {train_loss:.6e}  DATA_MSE {mse_val:.6e}")

    return model, logs
