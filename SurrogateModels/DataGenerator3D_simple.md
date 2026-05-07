# SIMPLE Formulation in `DataGenerator3D.py`

## 1. Scope

This note documents the three-dimensional SIMPLE algorithm implemented by
`DataGenerator3D.py` and `PressureUpdaters3D.py`.  The solver is written in a
dimensionless cylindrical passage coordinate system

\[
(R,\Theta,Z)\in[0,1]^3,
\]

where \(R\) is the normalized radial coordinate, \(\Theta\) is the normalized
periodic pitch coordinate, and \(Z\) is the normalized axial coordinate.  The
unknown fields are

\[
\mathbf{U}=(U_R,U_\Theta,U_Z), \qquad P,
\]

stored in tensors of shape \((n_R,n_\Theta,n_Z)\).  The implementation is
therefore a genuine 3D finite-volume method, not a stack of independent 2D
slices.

The derivation follows the notation and structure of `Dimensionless
Document.pdf`, especially the nondimensionalization in Section 1.1.2, the 3D
governing equations in Eqs. (2.49)-(2.52), the finite-volume linearization in
Eqs. (2.69)-(2.85), and the pressure-correction construction in Eqs.
(2.86)-(2.105).

## 2. Dimensionless Parameters

The code defines the main physical scales as

\[
u_\omega = r_s |\omega|,\qquad
u_{z0} = \frac{q_v}{\pi(r_s^2-r_h^2)},
\]

and the pressure scale

\[
P_0 = \rho\left(\frac{1}{2}u_\omega^2
      + \frac{1}{2}u_{z0}^2 + g h\right).
\]

The principal nondimensional groups used by the solver are

\[
Re_\omega = \frac{u_\omega (r_s-r_h)}{\nu},\qquad
Eu_\omega = \frac{P_0}{\rho u_\omega^2},
\]

\[
\Lambda = \frac{r_s-r_h}{h},\qquad
Ku = \frac{u_{z0}}{u_\omega},\qquad
\delta = \frac{r_s-r_h}{r_s}.
\]

The local cylindrical metric used in the pitch direction is

\[
K_\Theta(R) =
\frac{1}{\hat r\,\theta_0},\qquad
\hat r = R + \frac{r_h}{r_s-r_h},\qquad
\theta_0 = \frac{2\pi}{N_b}.
\]

The source term associated with gravity in the axial momentum equation is

\[
G^\star = \frac{g(r_s-r_h)}{u_\omega u_{z0}}.
\]

## 3. Finite-Volume Continuity Equation

For each cell, the discrete mass residual is assembled from the six face
fluxes

\[
D = F_E + F_W + F_N + F_S + F_T + F_B.
\]

The code computes these fluxes as

\[
F_E =
\frac{\hat r_E}{\hat r_C} A_R U_{R,E},
\qquad
F_W =
-\frac{\hat r_W}{\hat r_C} A_R U_{R,W},
\]

\[
F_N =
K_{\Theta,C} A_\Theta U_{\Theta,N},
\qquad
F_S =
-K_{\Theta,C} A_\Theta U_{\Theta,S},
\]

\[
F_T =
\Lambda Ku\, A_Z U_{Z,T},
\qquad
F_B =
-\Lambda Ku\, A_Z U_{Z,B}.
\]

Here \(E/W\) denote radial faces, \(N/S\) pitch-periodic faces, and \(T/B\)
axial top/bottom faces.  IBM blade blockage is introduced through face opening
factors

\[
\phi_f = \min(\phi_C,\phi_{nb}),
\]

so the residual used by the pressure equation is
is

\[
D_\phi =
\phi_E F_E + \phi_W F_W + \phi_N F_N + \phi_S F_S
+ \phi_T F_T + \phi_B F_B.
\]

The reported SIMPLE/COUPLE mass residual is the maximum finite-volume flux
imbalance over the fluid residual mask:

\[
\epsilon_m =
\max_{\Omega_f}
\left|D_\phi\right|.
\]

This is intentionally a finite-volume residual rather than \(D_\phi/\Delta V\).
The latter is useful as a local divergence-density diagnostic, but it scales
with grid refinement and can make a well-balanced fine-grid SIMPLE solution
appear non-convergent.

## 4. Momentum Linearization

The three momentum equations are linearized by an upwind finite-volume scheme.
For each component, the neighbor coefficients have the generic form

\[
a_f = D_f - \min(F_f,0),
\]

where \(D_f\) is a diffusive conductance and \(F_f\) is an oriented convective
flux.  The diagonal convective contribution is

\[
a_P^{conv} = \sum_f \max(F_f,0),
\]

and the total diffusive contribution is

\[
a_P^{diff} = D_E+D_W+D_N+D_S+D_T+D_B.
\]

The code uses a pseudo-transient stabilizing term

\[
a_P^{pseudo} = \frac{\Delta V}{\Delta \tau},
\]

optionally augmented by a local CFL-like spectral radius.  This term is not a
new physical term in the steady equations; it improves nonlinear stability of
the iterative solve.

The radial, tangential, and axial momentum diagonals are assembled as

\[
A_{11} =
a_P^{conv}+a_P^{diff}+a_P^{pseudo}+(1-\phi)
+ \frac{\Delta V}{Re_\omega \hat r^2},
\]

\[
A_{22} =
a_P^{conv}+a_P^{diff}+a_P^{pseudo}+(1-\phi)
+ \frac{\Delta V}{Re_\omega \hat r^2}
+ c_c\frac{\Delta V U_R}{\hat r},
\]

\[
A_{33} =
a_P^{conv}+a_P^{diff}+a_P^{pseudo}+(1-\phi).
\]

The \(R-\Theta\) local coupling is kept through

\[
A_{12},\quad A_{21},
\]

which represent geometric and rotating-frame coupling.  In code they are
limited so that

\[
A_{11}A_{22}-A_{12}A_{21}
\]

remains safely positive.  The local momentum block is therefore

\[
A =
\begin{bmatrix}
A_{11} & A_{12} & 0 \\
A_{21} & A_{22} & 0 \\
0      & 0      & A_{33}
\end{bmatrix}.
\]

This is the discrete counterpart of Eqs. (2.79)-(2.85).

## 5. Rhie-Chow Face Velocities

The solver uses a collocated grid.  To avoid pressure checkerboarding, face
velocities are not simple arithmetic averages.  Instead, the Rhie-Chow form is
used:

\[
\mathbf{U}_f =
\bar{\mathbf{U}}_f
-\left[
(A^{-1}\nabla P)_f
- \overline{A^{-1}\nabla P}_f
\right].
\]

For the \(R-\Theta\) block,

\[
A^{-1}_{R\Theta}
=
\frac{1}{\det A_{R\Theta}}
\begin{bmatrix}
A_{22} & -A_{12} \\
-A_{21} & A_{11}
\end{bmatrix},
\]

\[
\det A_{R\Theta}=A_{11}A_{22}-A_{12}A_{21}.
\]

The axial inverse is simply

\[
A^{-1}_{ZZ} = \frac{1}{A_{33}}.
\]

This corresponds to Eqs. (2.93)-(2.96).  The important point is that the code
does not use a Cartesian diagonal approximation \(1/a_P\) for all velocity
components; it preserves the local \(R-\Theta\) coupling before constructing
pressure correction fluxes.

## 6. SIMPLE Pressure Correction

The SIMPLE step uses the standard decomposition

\[
\mathbf{U} = \mathbf{U}^\ast + \mathbf{U}',
\qquad
P = P^\ast + P',
\]

where \(\mathbf{U}^\ast\) is the momentum predictor.  The pressure correction
induces the velocity correction

\[
\mathbf{U}' = -A^{-1}G(P'),
\]

where \(G(P')\) is the discrete pressure-gradient vector:

\[
G_R(P') =
\Delta V\, Eu_\omega\, \frac{\partial P'}{\partial R},
\]

\[
G_\Theta(P') =
\Delta V\, Eu_\omega\, K_\Theta
\frac{\partial P'}{\partial \Theta},
\]

\[
G_Z(P') =
\Delta V\, Eu_\omega\, \frac{\Lambda}{Ku}
\frac{\partial P'}{\partial Z}.
\]

Substituting \(\mathbf{U}'\) into continuity gives the pressure-correction
equation

\[
\mathcal{L}(P') = -D_\phi(\mathbf{U}^\ast,P^\ast),
\]

where \(\mathcal{L}\) is assembled by `PressureUpdater3D.coefficients`.

The direct face coefficients are derived from Eqs. (2.98)-(2.100).  The mixed
\(R-\Theta\) response introduces corner terms

\[
NE,\quad NW,\quad SE,\quad SW,
\]

which correspond to the mixed derivative contributions in Eqs. (2.103)-(2.104).
The operator therefore has the form

\[
\begin{aligned}
\mathcal{L}(P')_C
=&\ C P'_C
- E P'_E - W P'_W
- N P'_N - S P'_S \\
&- T P'_T - B P'_B
- NE P'_{NE} - NW P'_{NW}
- SE P'_{SE} - SW P'_{SW}.
\end{aligned}
\]

Boundary pressure correction is projected as

\[
P'=0
\]

at the inlet and outlet pressure boundaries.  Radial walls have no
pressure-correction flux through the physical wall faces.

After solving for \(P'\), the code updates

\[
\mathbf{U}^{new}
=
\mathbf{U}^{old}
+ \alpha_U\left(\mathbf{U}^{\ast}+\mathbf{U}'-\mathbf{U}^{old}\right),
\]

\[
P^{new}=P^{old}+\alpha_P P',
\]

where \(\alpha_U\) is `u_relax` and \(\alpha_P\) is `p_relax`.

## 7. Algorithmic Structure

One SIMPLE inner iteration in `simple_step()` is:

1. Assemble convective, diffusive, pseudo-transient, and local coupling
   coefficients.
2. Solve the three momentum predictor equations for
   \((U_R^\ast,U_\Theta^\ast,U_Z^\ast)\).
3. Assemble and solve the pressure-correction equation.
4. Correct velocity and pressure.
5. Re-apply velocity, pressure, periodic, IBM, and flow-boundary projections.
6. Evaluate momentum and continuity residuals.

The outer loop in `solve(method="simple")` is not a second pressure iteration.
It is a flow-control loop.  After a block of SIMPLE iterations at fixed
\(\Delta P\), the solver computes

\[
q_{\text{hat}} =
\frac{\int_{\Gamma_{out}}\phi U_Z\,dA}
     {\int_{\Gamma_{out}}\phi\,dA}.
\]

If \(q_{\text{hat}}\neq 1\), the scalar pressure drop
`delta_p_global` is nudged and the inner SIMPLE solve continues.  Thus:

- the pressure-correction iterations solve local continuity for a fixed
  pressure-drop condition;
- the outer loop adjusts the global pressure drop to meet the target flow rate.

## 8. Relation to COUPLE Mode

The COUPLE mode reuses the same momentum coefficients, Rhie-Chow fluxes, and
pressure-correction operator.  The difference is algorithmic: COUPLE applies a
pseudo-transient velocity update and then projects pressure with backtracking
and update limits.  The mass residual in COUPLE is therefore controlled by how
much of the pressure projection is accepted.

In the current implementation, pressure projection acceptance is based mainly
on continuity reduction, with a bounded allowance for temporary growth in the
momentum residual.  This is necessary because an effective projection can
substantially reduce divergence while temporarily increasing the nonlinear
momentum residual.  A criterion that rejects any momentum growth almost
completely suppresses pressure projection and causes the mass residual to
stagnate.

## 9. Implementation Map

The principal implementation locations are:

- `DataGenerator3D._face_fluxes`: finite-volume fluxes.
- `DataGenerator3D._face_opening`: IBM face opening factors.
- `DataGenerator3D._assemble_momentum_coefficients`: upwind momentum
  coefficients and local block entries.
- `DataGenerator3D.rhie_chow`: collocated-grid face velocity interpolation.
- `PressureUpdater3D.inverse_momentum_blocks`: local inverse of the
  \(R-\Theta\) block and axial diagonal.
- `PressureUpdater3D.coefficients`: pressure-correction stencil coefficients.
- `PressureUpdater3D.velocity_correction`: computation of
  \(\mathbf{U}'=-A^{-1}G(P')\).
- `BladeCalc3D.simple_step`: one complete SIMPLE iteration.
- `BladeCalc3D.solve`: outer flow-control loop around SIMPLE.

## 10. Practical Notes

1. The reported mass residual is a maximum finite-volume flux imbalance.  A
   divergence-density diagnostic \(D_\phi/\Delta V\) is more stringent but is
   grid-size sensitive and should not be used as the primary convergence flag.
2. Fixed flow-boundary mode enforces a target weighted axial profile on the
   physical inlet/outlet faces.  The adjacent velocity cells keep zero-gradient
   values so that SIMPLE pressure corrections are not overwritten after the
   pressure equation has balanced the internal Rhie-Chow faces.
3. If `pressure_solver="gmg"` is requested but mixed corner coefficients are
   nonzero, `PressureUpdater3D` falls back to BiCGStab, because the full operator
   is not a pure six-point stencil.
4. The pseudo-transient terms and update limits are numerical stabilization
   devices.  They are not part of the steady nondimensional equations, but they
   are important for robust nonlinear convergence in the generated blade
   passages.
