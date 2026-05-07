# Methodological Documentation for the Blade Flow Surrogate Model

## Abstract

This document summarizes the complete surrogate modeling workflow implemented in this repository. The model maps blade geometry, operating conditions, and immersed-boundary descriptors to three-dimensional cylindrical flow fields. The implementation combines slice-wise neural operators, dimensionless physics residuals, adaptive immersed-boundary smoothing, optional function-space KKT projection, and post-processing utilities for physical inspection. The document is written as a technical description of the current code base, with particular attention to the loss functions, training strategy, input and output design, the full inventory of `NeuralOperators.py`, and the role of `KKTProjectionOperators.py`.

## 1. Problem Setting

The surrogate model approximates a mapping from a parametrized rotating blade passage and operating condition to a dimensionless flow field on a structured cylindrical grid. For each case, the grid is indexed by normalized radial, circumferential, and axial coordinates:

```text
R in [0, 1], Theta in [0, 1], Z in [0, 1].
```

The target flow variables are

```text
UR(R, Theta, Z), UT(R, Theta, Z), UZ(R, Theta, Z), P(R, Theta, Z).
```

The first three variables denote radial, tangential, and axial velocities, while `P` denotes pressure. Internally, training is performed in a dimensionless representation. Physical fields are recovered through the characteristic scales defined in `FlowCaseConfig`, namely `u_omega`, `u_zo`, and `P0`.

The model can operate in two regimes:

1. Supervised surrogate learning, when simulated labels `UR`, `UT`, `UZ`, and `P` are available.
2. Pure physics debugging, when no labels are provided and the model is trained only through physical residuals and integral constraints.

## 2. Case Configuration and Dimensionless Groups

Each sample is represented by a `FlowCaseConfig` object. The required physical and geometric quantities include:

```text
n, rh, rs, h, mu, rho, omega, qv, n_blade
```

Optional fields include:

```text
z0, g, g_star, ibm_C, ibm_epsilon, absolute_frame, use_absolute_omega_scale
```

The implementation constructs several derived scales and dimensionless groups:

```text
delta_r      = rs - rh
theta0       = 2*pi/n_blade
u_omega      = |rs*omega|, unless signed scaling is requested
u_zo         = qv/(pi*(rs^2 - rh^2))
P0           = rho*(u_omega^2/2 + u_zo^2/2 + g*h)
Re_omega     = u_omega*delta_r/(mu/rho)
Eu_omega     = P0/(rho*u_omega^2)
Lambda       = delta_r/h
Ku           = u_zo/u_omega
delta        = delta_r/rs
sgn_omega    = sign(omega)
qv_passage   = qv/n_blade
qv_hat       = qv_passage/(|u_zo|*delta_r^2*theta0)
```

These quantities are injected into the neural-operator input as spatially broadcast channels or scalar tensors used by the physical residuals.

## 3. Geometry, Immersed Boundary Representation, and Runtime Phi Reconstruction

The blade geometry can be provided in three equivalent forms:

1. `blade_params`: a blade parameter file used to reconstruct the blade boundary.
2. `blade_mask`: a sharp solid mask.
3. `phi`: a smooth immersed-boundary indicator.

If `blade_params` is available, `build_blade_boundary` is used to construct the sharp mask and signed distance field. If only `phi` is available, the mask is inferred from the threshold `phi < 0.5`.

The smooth immersed-boundary function is computed from signed distance `d` as

```text
phi = 1 - exp(-C(r)*d^2/epsilon(r)^2).
```

The implementation supports span-wise immersed-boundary profiles:

```text
C(r), epsilon(r).
```

The accepted case keys include:

```text
ibm_C_profile, ibm_C_span, ibm_C_r, ibm_C
ibm_epsilon_profile, ibm_epsilon_span, ibm_epsilon_r, ibm_epsilon
```

Scalar values are automatically expanded to length-`n` span profiles. Nonmatching profiles are linearly interpolated to the current radial resolution. During training and deployment, `SurrogateModeling._prepare_runtime_batch` recomputes `phi` from the current learned or stored profiles, then rebuilds the input tensor. This design makes the immersed-boundary transition trainable while keeping the rest of the geometry and operating-condition channels fixed.

## 4. Input and Output Design

The model input has shape

```text
[B, C, R, Theta, Z].
```

The first one or two channels encode the blade field, depending on `input_mode`:

```text
mask: blade_mask
phi:  phi
both: blade_mask and phi
```

For `input_mode="both"`, the current default input has 15 channels:

```text
blade_mask
phi
r_hat
K_theta
Theta_sin
Theta_cos
Z
solid_ut
Eu_omega
Re_omega
Lambda
Ku
delta
sgn_omega
g_star
```

Here

```text
r_hat    = R + rh/(rs-rh)
K_theta  = 1/(r_hat*theta0)
solid_ut = delta*r_hat
```

The output is a dictionary containing four dimensionless fields:

```text
UR, UT, UZ, P
```

with shape

```text
[B, R, Theta, Z].
```

The pressure is fixed by subtracting the reference value at `(R, Theta, Z) = (0, 0, 0)`.

## 5. Slice-Wise Neural Operator Architecture

The main model class is `SliceWiseFNOFlowModel`. Rather than using a fully three-dimensional neural operator, it applies a two-dimensional operator independently to each radial slice. The input tensor

```text
[B, C, R, Theta, Z]
```

is reshaped to

```text
[B*R, C, Theta, Z].
```

This representation assumes that every radial layer is a two-dimensional `Theta-Z` plane, while all radial layers share the same operator weights. The design is computationally lighter than a full 3D operator and exploits the strong slice structure of the passage geometry.

Because `Theta` is periodic but `Z` is not, the model applies a seam projection along `Theta` and replicate padding along `Z` before the two-dimensional operator is called. After prediction, the output is reshaped back to `[B, 4, R, Theta, Z]`.

The model applies several hard output constraints:

```text
UR = phi*UR_raw
UT = phi*UT_raw + (1 - phi)*solid_ut
UZ = phi*UZ_raw
```

Thus the solid region has no radial or axial velocity and follows the prescribed solid tangential velocity. Hub and shroud constraints are also enforced:

```text
UR(R=0) = UR(R=1) = 0
UZ(R=0) = UZ(R=1) = 0
UT(R=0) = solid_ut(R=0)
UT(R=1) = solid_ut(R=1)
```

All output fields are projected to be consistent across the periodic `Theta` seam.

## 6. Complete Inventory of NeuralOperators.py

The file `NeuralOperators.py` contains the reusable neural-operator components and a small Poisson demonstration utility. This section enumerates its complete structure.

### 6.1 Reproducibility

`seed_everything(seed)` sets deterministic random seeds for Python, NumPy, PyTorch CPU, and PyTorch CUDA. It also fixes cuDNN deterministic behavior.

### 6.2 Discrete Cosine Transform Utilities

The file implements DCT and inverse DCT utilities:

```text
dct_1d(x)
idct_1d(X)
dct_2d(x)
idct_2d(X)
```

These functions implement separable real-valued cosine transforms through FFT operations. They are used by Chebyshev spectral convolution, where low-order Chebyshev coefficients are represented through DCT coefficients.

### 6.3 Chebyshev Spectral Convolution

`ChebSpectralConv2d` maps

```text
[B, C_in, H, W] -> [B, C_out, H, W].
```

It applies a 2D DCT to the input, retains the top-left Chebyshev modes, multiplies them by learned real-valued spectral weights, zero-fills the remaining coefficients, and transforms back with the inverse DCT. It is suitable for nonperiodic directions but, like a standard truncated spectral operator, it is biased toward smooth low-frequency content.

### 6.4 Fourier Spectral Convolution

`SpectralConv2d` is the standard FNO spectral layer. It applies `torch.fft.rfft2`, retains a square block of low-frequency modes, multiplies them by learned complex weights, and reconstructs the field with `irfft2`. Its input and output shapes are

```text
[B, C_in, H, W] -> [B, C_out, H, W].
```

In the original implementation, only the low positive-frequency block is retained. This is efficient but can underrepresent sharp immersed-boundary features and high-frequency near-wall structures.

### 6.5 Mixed Boundary Padding

`mixed_boundary_pad2d(x, pad_h, pad_w)` pads the first spatial direction periodically and the second spatial direction by replication. This matches the model's `Theta-Z` interpretation, where `Theta` is periodic while `Z` is nonperiodic.

### 6.6 Multi-Band Spectral Convolution

`MultiBandSpectralConv2d` was introduced to improve high-frequency expressivity. It retains:

1. Standard low Fourier modes.
2. Negative `Theta` frequency rows in the full FFT plane.
3. A small high-frequency band along the rFFT `Z` direction.

This module still operates in Fourier space, but it does not restrict the representation exclusively to the lowest positive block. It is intended to reduce the tendency of the operator to learn only overly smooth fields.

### 6.7 Fourier Feature Grid

`FourierFeatureGrid2d` appends fixed coordinate features to the input:

```text
sin(2*pi*k*Theta), cos(2*pi*k*Theta),
sin(2*pi*k*Z),     cos(2*pi*k*Z).
```

The default bands are

```text
(1, 2, 4, 8).
```

These features give the pointwise lifting MLP direct access to multiple spatial frequencies. This reduces the burden on the spectral trunk to synthesize all oscillatory content from low-mode coefficients.

### 6.8 Local High-Pass Block

`LocalHighPassBlock2d` computes a local high-pass residual by subtracting a local average from the feature map. It then applies depthwise and pointwise convolutions. Its purpose is to capture edges, short-wavelength features, and immersed-boundary transition layers that a low-mode spectral layer tends to smooth out.

### 6.9 High-Frequency CFNO Block

`HFCFNOBlock` combines three branches:

1. A low-frequency `CFNOBlock`.
2. A `MultiBandSpectralConv2d` branch.
3. A `LocalHighPassBlock2d` branch.

The high-frequency branches are gated by a learned scalar. This allows the network to begin from a relatively conservative low-frequency representation and activate high-frequency corrections during optimization.

### 6.10 CFNO Block

`CFNOBlock` combines a Fourier spectral convolution and a Chebyshev spectral convolution. It learns a sigmoid blending parameter `alpha` and also fuses the two branch outputs through a pointwise convolution. The output is

```text
alpha*Fourier(x) + (1-alpha)*Chebyshev(x) + fuse([Fourier(x), Chebyshev(x)]).
```

This block is the core of the legacy CFNO models.

### 6.11 CFNO2d

`CFNO2d` is an illustrative two-channel CFNO network. It lifts a two-channel input to width `width`, applies repeated CFNO blocks with pointwise residual convolutions, and projects the result back to two output channels. It is retained as an example stack.

### 6.12 FNO2d_small

`FNO2d_small` is the compact Fourier neural operator used by older checkpoints. It consists of:

1. A pointwise lifting layer `fc0`.
2. A stack of `SpectralConv2d` layers.
3. Pointwise convolutional residuals.
4. A pointwise MLP head `fc1 -> fc2`.

It supports configurable `input_features` and `output_features`, which allows it to be used as the slice operator inside `SliceWiseFNOFlowModel`.

### 6.13 CNO2d_small

`CNO2d_small` mirrors `FNO2d_small` but replaces Fourier spectral blocks with `ChebSpectralConv2d` blocks. It is useful when nonperiodic or cosine-basis behavior is preferred.

### 6.14 CFNO2d_small

`CFNO2d_small` uses `CFNOBlock` in the same compact architecture pattern as `FNO2d_small` and `CNO2d_small`. It was the previous default operator core before the high-frequency extension.

### 6.15 HF_CFNO2d_small

`HF_CFNO2d_small` is the current default operator core for new training runs. It has the following structure:

```text
input
  -> FourierFeatureGrid2d
  -> pointwise lifting fc0
  -> repeated HFCFNOBlock + 1x1 residual convolution
  -> pointwise MLP head
  -> output
```

It is designed to preserve the smooth global modeling capacity of CFNO while adding explicit mechanisms for high-frequency and local boundary features.

### 6.16 Poisson Demonstration Utilities

The file also contains a small Poisson example:

```text
poisson(f, iters, tol)
make_dataset(...)
laplacian(u, dx, dy)
train_model(...)
```

These utilities generate synthetic Poisson data with Gaussian sources and train simple operator models with data and PDE residual terms. They are not the main blade-flow training path, but they provide a compact reference for physics-informed operator training.

## 7. Adaptive Span-Wise IBM Parameter Head

`AdaptiveIBMMaskController` learns span-wise profiles `C(r)` and `epsilon(r)`. Its features are computed per radial layer and include:

```text
span_r
blade_fraction(r)
mean(|signed_distance|)(r)
std(|signed_distance|)(r)
qv_hat
log(Re_omega)
log(Eu_omega)
Lambda
delta
```

The MLP outputs two raw values per span. These are mapped into prescribed ranges:

```text
C(r)       in ibm_c_range
epsilon(r) in ibm_epsilon_range
```

The final shape is

```text
[B, R, 1, 1],
```

which broadcasts naturally over `Theta` and `Z` when reconstructing `phi`.

The head is initialized near the case-default `C` and `epsilon` values. This prevents the initial optimization from creating an excessively distorted immersed-boundary transition layer.

## 8. Supervised Loss

If target labels are present, the supervised loss compares the predicted dimensionless fields with normalized target fields. If no labels are present, the supervised loss returns zero and the sample is treated as pure physics data.

The weighted mean squared error is implemented as

```text
WMSE(r, w) = sum(w*r^2)/max(sum(w), eps).
```

The velocity losses are computed over the full sample when `has_target=1`:

```text
L_UR = WMSE(UR_pred - UR_true, has_target)
L_UT = WMSE(UT_pred - UT_true, has_target)
L_UZ = WMSE(UZ_pred - UZ_true, has_target)
```

The pressure loss is weighted by the fluid indicator:

```text
L_P = WMSE(P_pred - P_true, phi*has_target).
```

The supervised data loss is

```text
L_data = L_UR + L_UT + L_UZ + L_P.
```

Pressure targets are normalized by subtracting `P[0,0,0]`, consistent with the pressure reference used in the network output.

## 9. Physics Loss

The physical loss is implemented in `BladeFlowPhysicsLoss`. It contains five trainable residual terms:

```text
L_phys = L_c + L_r + L_theta + L_z + L_qv.
```

The residuals are evaluated in the fluid region using a PDE mask derived from `phi`, with radial and axial boundary layers excluded from the PDE residual. The periodic `Theta` endpoint is handled by a special duplicate-endpoint finite difference.

### 9.1 Continuity Residual

The dimensionless continuity residual is

```text
R_c =
  (1/r_hat) d_R(r_hat*UR)
  + K_theta d_Theta(UT)
  + Lambda*Ku d_Z(UZ).
```

The corresponding loss is

```text
L_c = WMSE(R_c, pde_mask).
```

### 9.2 Dimensionless Laplacian

For each velocity component, the cylindrical dimensionless Laplacian is

```text
Lap(f) =
  d_RR(f)
  + (1/r_hat) d_R(f)
  + K_theta^2 d_ThetaTheta(f)
  + Lambda^2 d_ZZ(f).
```

This operator is used in the three momentum residuals.

### 9.3 Radial Momentum Residual

The radial residual is

```text
R_r =
  UR d_R(UR)
  + K_theta UT d_Theta(UR)
  + Lambda Ku UZ d_Z(UR)
  - UT^2/r_hat
  + Eu d_R(P)
  - ( Lap(UR) - UR/r_hat^2 - (2 K_theta/r_hat)d_Theta(UT) )/Re.
```

If `absolute_frame=False`, additional rotating-frame source terms are added.

### 9.4 Tangential Momentum Residual

The tangential residual is

```text
R_theta =
  UR d_R(UT)
  + K_theta UT d_Theta(UT)
  + Lambda Ku UZ d_Z(UT)
  + UR UT/r_hat
  + K_theta Eu d_Theta(P)
  - ( Lap(UT) - UT/r_hat^2 + (2 K_theta/r_hat)d_Theta(UR) )/Re.
```

Rotating-frame terms are added when requested.

### 9.5 Axial Momentum Residual

The axial residual is

```text
R_z =
  UR d_R(UZ)
  + K_theta UT d_Theta(UZ)
  + Lambda Ku UZ d_Z(UZ)
  + (Lambda/Ku) Eu d_Z(P)
  - Lap(UZ)/Re
  + g_star.
```

### 9.6 Outlet Flow-Rate Constraint

The outlet flow-rate residual compares the predicted single-passage dimensionless flow rate against `qv_hat`:

```text
L_qv = mean((q_hat_pred - qv_hat)^2).
```

This term is essential in pure physics mode because it anchors the axial flow magnitude.

### 9.7 Diagnostic Boundary Terms

Two additional diagnostics are computed:

```text
loss_bc_periodic
loss_bc_blade
```

They are not included in `L_phys` by default because the forward pass already enforces periodic seam consistency and solid-region velocity constraints. They are retained for monitoring whether the hard constraints remain effective.

## 10. Training Objective and Optimization Strategy

The total training loss in an epoch is

```text
L_total = data_weight*L_data + physics_factor(epoch)*L_phys.
```

The physics factor follows a warmup and ramp schedule:

```text
physics_factor = 0,                                  epoch < warmup_epochs
physics_factor = physics_weight*ramp_fraction,       during ramp
physics_factor = physics_weight,                     after ramp
```

In pure physics debugging mode, `data_weight=0`, `physics_weight=1`, and the warmup/ramp values are zero. In supervised mode, the ramp allows the model to first learn the data scale before the PDE residual is made dominant.

The optimizer is Adam. The optimized parameters include:

1. The slice-wise neural operator.
2. The adaptive IBM parameter controller, if enabled.

The KKT projection module has no trainable parameters. It is inserted into the computational graph as a differentiable operation.

## 11. Runtime Training Loop

For each batch, the training loop performs the following sequence:

1. Move tensors to the selected device.
2. Recompute `C(r)` and `epsilon(r)` if adaptive IBM learning is enabled.
3. Reconstruct `phi` from signed distance and span-wise IBM profiles.
4. Recompose the neural-operator input tensor.
5. Apply the slice-wise neural operator.
6. Enforce hard velocity and periodic output constraints.
7. Optionally apply the KKT projection layer.
8. Compute supervised and physics losses.
9. Backpropagate and update trainable parameters.

This ordering is important: both the adaptive IBM reconstruction and the optional KKT projection are inside the training graph, not post-processing steps.

## 12. KKT Projection Layer

The file `KKTProjectionOperators.py` introduces a function-space analogue of the KKT projection layer discussed in hard-constrained PINN literature. The finite-dimensional projection

```text
y_star = y_hat - B^T (B B^T)^(-1) (B y_hat + A x - b)
```

is generalized to a field projection:

```text
u_star = u_hat - C^* lambda,
C C^* lambda = C(u_hat).
```

For the current implementation, `C` is the cylindrical divergence operator used by the continuity residual:

```text
C(u) =
  (1/r_hat) d_R(r_hat*UR)
  + K_theta d_Theta(UT)
  + Lambda*Ku d_Z(UZ).
```

The projection is implemented in `CylindricalDivergenceKKTProjection`. It provides:

```text
divergence(UR, UT, UZ, batch)
gradient(lambda, batch)
normal_operator(lambda, batch)
solve_multiplier(rhs, batch, weight)
apply_hard_velocity_constraints(...)
forward(pred, batch)
```

The multiplier `lambda` is solved by a matrix-free conjugate-gradient style iteration. The velocity correction is

```text
UR_star = UR_hat - strength*d_R(lambda)
UT_star = UT_hat - strength*K_theta*d_Theta(lambda)
UZ_star = UZ_hat - strength*Lambda*Ku*d_Z(lambda).
```

After correction, the blade, hub, and shroud hard velocity constraints are re-applied. The projection returns diagnostic quantities:

```text
kkt_divergence_before
kkt_divergence_after
```

The projection is optional and controlled by:

```text
use_kkt_projection
kkt_projection_iters
kkt_projection_strength
```

It is currently a relaxed projection with default strength `0.35`. This is intentional: a full projection can overcorrect on coarse grids or near immersed-boundary discontinuities unless the boundary treatment and normal equation are tuned carefully.

## 13. Checkpoint Compatibility

New training runs default to

```text
operator_variant = "hf_cfno".
```

The code can also instantiate:

```text
"fno"
"cfno"
"hf_cfno"
```

Old checkpoints are handled by inspecting the state-dict keys. If legacy `core.blocks.*.weights` entries are found, the model is reconstructed with `FNO2d_small`. Otherwise, legacy checkpoints without an explicit `operator_variant` are interpreted as `CFNO2d_small`. This preserves older model functionality while allowing newer runs to use the high-frequency operator.

## 14. Interpretation of the High-Frequency Upgrade

The original FNO and CFNO operators primarily retain low spectral modes. This is a strong smoothness prior and is often beneficial for stable operator learning. However, the blade-flow surrogate must represent:

1. Sharp immersed-boundary transitions.
2. Near-wall velocity gradients.
3. Wake-like short-wavelength structures.
4. Local variations induced by span-wise blade geometry.

Without sufficient data supervision, a purely low-frequency operator can minimize coarse residuals while failing to represent these high-frequency structures. The high-frequency CFNO variant addresses this by augmenting the spectral basis, adding coordinate Fourier features, and introducing a local high-pass branch. It should therefore be interpreted as an expressivity upgrade rather than a replacement for physics-informed training.

## 15. Recommended Experimental Protocol

A controlled comparison should use the following model variants:

```text
A. operator_variant="fno",     use_kkt_projection=False
B. operator_variant="cfno",    use_kkt_projection=False
C. operator_variant="hf_cfno", use_kkt_projection=False
D. operator_variant="hf_cfno", use_kkt_projection=True
```

For the KKT projection variant, start with:

```text
kkt_projection_iters    = 12 to 24
kkt_projection_strength = 0.2 to 0.5
```

The evaluation should not rely only on `loss_total`. The following metrics should be monitored independently:

```text
loss_c
loss_r
loss_theta
loss_z
loss_qv
loss_bc_blade
kkt_div_before
kkt_div_after
q_hat_pred
q_hat_target
```

In addition, span-wise visual inspection is necessary because high-frequency expressivity can improve local structure while also increasing oscillatory artifacts if the physical residuals are underweighted.

## 16. Summary

The current surrogate model is a dimensionless, slice-wise neural-operator framework for blade-passage flow prediction. Its default architecture is now a high-frequency CFNO designed to improve representation of immersed-boundary and near-wall features. The training objective combines supervised data loss, cylindrical Navier-Stokes residuals, an outlet flow-rate constraint, and hard output constraints. Adaptive span-wise IBM profiles allow the smooth boundary transition to be learned rather than fixed. The optional KKT projection layer introduces a differentiable function-space projection that acts on the velocity field before loss evaluation. Together, these components form a hybrid operator-learning framework in which neural expressivity, physical residuals, and hard-constraint projection are coupled within the training graph.
