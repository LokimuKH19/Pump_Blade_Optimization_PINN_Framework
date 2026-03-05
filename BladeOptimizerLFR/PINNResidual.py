"""PINN residual utilities for non-dimensional rotating-frame pump flow.

This module implements residuals on a regular computational cube
(R, Theta, Z) in [0, 1]^3, with an immersed-boundary style solid mask.

Key modeling choices (matching the current discussion):
- Solve on a structured unit cube, keep periodicity in Theta.
- Use non-dimensional variables U_R, U_Theta, U_Z, P.
- Treat blade as immersed solid through a mask phi(R,Theta,Z):
    phi=1 in fluid, phi=0 in solid.
  Solid hard constraints can be enforced by multiplying outputs with phi
  and pinning pressure to a constant in solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class NondimParams:
    """Non-dimensional and geometric constants used in residuals.

    Geometry:
        Rh: hub radius
        Rs: shroud radius
        H: axial height
        N: blade count per periodic sector mapping

    Scaling:
        omega: rotor angular speed (signed)
        uz_ref: reference axial speed (e.g. outlet bulk speed)
        rho: density
        nu: kinematic viscosity
        p_scale: pressure scale P0 used in P=(p-p_i)/P0
        g: gravity acceleration
    """

    Rh: float
    Rs: float
    H: float
    N: int

    omega: float
    uz_ref: float
    rho: float
    nu: float
    p_scale: float
    g: float = 9.81

    @property
    def dR(self) -> float:
        return self.Rs - self.Rh

    @property
    def sign_omega(self) -> float:
        if self.omega > 0:
            return 1.0
        if self.omega < 0:
            return -1.0
        return 0.0

    @property
    def U_omega(self) -> float:
        return abs(self.omega) * self.Rs

    @property
    def eta(self) -> float:
        # r/dR = R + eta
        return self.Rh / self.dR

    @property
    def Lambda(self) -> float:
        # dR / H
        return self.dR / self.H

    @property
    def Kz(self) -> float:
        # uz_ref / U_omega
        eps = 1e-12
        return self.uz_ref / (self.U_omega + eps)

    @property
    def Eu(self) -> float:
        # Euler-like pressure ratio: P0/(rho*U_omega^2)
        eps = 1e-12
        return self.p_scale / (self.rho * (self.U_omega**2 + eps))

    @property
    def Re_omega(self) -> float:
        # rotational Reynolds number based on dR and U_omega
        eps = 1e-12
        return self.U_omega * self.dR / (self.nu + eps)

    @property
    def Gstar(self) -> float:
        # gravity in axial equation scaled by U_omega*uz_ref/dR
        eps = 1e-12
        return self.g * self.dR / (self.U_omega * self.uz_ref + eps)


@dataclass
class PINNFields:
    """Predicted fields at collocation points."""

    U_R: torch.Tensor
    U_Theta: torch.Tensor
    U_Z: torch.Tensor
    P: torch.Tensor


@dataclass
class ResidualPack:
    continuity: torch.Tensor
    mom_r: torch.Tensor
    mom_theta: torch.Tensor
    mom_z: torch.Tensor


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def _second(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _grad(_grad(y, x), x)


def immersed_hard_constraint(
    raw_fields: PINNFields,
    phi: torch.Tensor,
    p_solid_const: float = 0.0,
) -> PINNFields:
    """Apply immersed-boundary style hard constraints.

    phi in [0,1]:
      - fluid region phi=1: keep predictions
      - solid region phi=0: velocity -> 0, pressure -> constant
    """

    p0 = torch.as_tensor(p_solid_const, dtype=raw_fields.P.dtype, device=raw_fields.P.device)
    return PINNFields(
        U_R=phi * raw_fields.U_R,
        U_Theta=phi * raw_fields.U_Theta,
        U_Z=phi * raw_fields.U_Z,
        P=phi * raw_fields.P + (1.0 - phi) * p0,
    )


def compute_residuals(
    R: torch.Tensor,
    Theta: torch.Tensor,
    Z: torch.Tensor,
    fields: PINNFields,
    prm: NondimParams,
    phi: Optional[torch.Tensor] = None,
) -> ResidualPack:
    """Compute non-dimensional residuals on unit cube coordinates.

    Coordinates:
        R, Theta, Z in [0,1], each tensor shape [N,1] (or broadcast-compatible).

    Notes:
    - If phi is provided, PDE residuals are multiplied by phi so only fluid points
      contribute (immersed boundary style domain masking).
    - Theta-periodicity is handled by dedicated boundary losses, not here.
    """

    U_R, U_T, U_Z, P = fields.U_R, fields.U_Theta, fields.U_Z, fields.P

    eta = prm.eta
    lam = prm.Lambda
    kz = prm.Kz
    eu = prm.Eu
    inv_re = 1.0 / (prm.Re_omega + 1e-12)
    sgn = prm.sign_omega
    delta = prm.dR / prm.Rs

    rhat = R + eta  # r / dR
    ktheta = prm.N / (2.0 * torch.pi * rhat)

    # first derivatives
    UR_R = _grad(U_R, R)
    UR_T = _grad(U_R, Theta)
    UR_Z = _grad(U_R, Z)

    UT_R = _grad(U_T, R)
    UT_T = _grad(U_T, Theta)
    UT_Z = _grad(U_T, Z)

    UZ_R = _grad(U_Z, R)
    UZ_T = _grad(U_Z, Theta)
    UZ_Z = _grad(U_Z, Z)

    P_R = _grad(P, R)
    P_T = _grad(P, Theta)
    P_Z = _grad(P, Z)

    # second derivatives
    UR_RR = _second(U_R, R)
    UR_TT = _second(U_R, Theta)
    UR_ZZ = _second(U_R, Z)

    UT_RR = _second(U_T, R)
    UT_TT = _second(U_T, Theta)
    UT_ZZ = _second(U_T, Z)

    UZ_RR = _second(U_Z, R)
    UZ_TT = _second(U_Z, Theta)
    UZ_ZZ = _second(U_Z, Z)

    # cylindrical Laplacian in (R,Theta,Z) mapped coordinates
    lap_UR = UR_RR + (1.0 / rhat) * UR_R + (ktheta**2) * UR_TT + (lam**2) * UR_ZZ
    lap_UT = UT_RR + (1.0 / rhat) * UT_R + (ktheta**2) * UT_TT + (lam**2) * UT_ZZ
    lap_UZ = UZ_RR + (1.0 / rhat) * UZ_R + (ktheta**2) * UZ_TT + (lam**2) * UZ_ZZ

    # continuity:
    # (1/rhat)d_R(rhat*U_R) + N/(2pi rhat)d_Theta(U_Theta) + kz*lam*d_Z(U_Z) = 0
    continuity = (1.0 / rhat) * _grad(rhat * U_R, R) + ktheta * UT_T + kz * lam * UZ_Z

    # radial momentum (scaled by U_omega^2 / dR)
    mom_r = (
        U_R * UR_R
        + ktheta * U_T * UR_T
        + kz * lam * U_Z * UR_Z
        - (U_T**2) / rhat
        + eu * P_R
        - inv_re * (lap_UR - U_R / (rhat**2) - 2.0 * ktheta / rhat * UT_T)
        - (delta**2) * rhat
        + 2.0 * sgn * delta * U_T
    )

    # azimuthal momentum (scaled by U_omega^2 / dR)
    mom_theta = (
        U_R * UT_R
        + ktheta * U_T * UT_T
        + kz * lam * U_Z * UT_Z
        + (U_R * U_T) / rhat
        + eu * ktheta * P_T
        - inv_re * (lap_UT - U_T / (rhat**2) + 2.0 * ktheta / rhat * UR_T)
        - 2.0 * sgn * delta * U_R
    )

    # axial momentum (scaled by U_omega*uz_ref / dR)
    mom_z = (
        U_R * UZ_R
        + ktheta * U_T * UZ_T
        + kz * lam * U_Z * UZ_Z
        + eu * (lam / (kz + 1e-12)) * P_Z
        - inv_re * lap_UZ
        + prm.Gstar
    )

    if phi is not None:
        continuity = phi * continuity
        mom_r = phi * mom_r
        mom_theta = phi * mom_theta
        mom_z = phi * mom_z

    return ResidualPack(
        continuity=continuity,
        mom_r=mom_r,
        mom_theta=mom_theta,
        mom_z=mom_z,
    )


def periodic_mismatch(
    fields_lo: PINNFields,
    fields_hi: PINNFields,
    match_gradients: bool = False,
    lo_theta: Optional[torch.Tensor] = None,
    hi_theta: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Return Theta-periodic mismatches at Theta=0 and Theta=1 boundaries."""

    out = {
        "U_R": fields_lo.U_R - fields_hi.U_R,
        "U_Theta": fields_lo.U_Theta - fields_hi.U_Theta,
        "U_Z": fields_lo.U_Z - fields_hi.U_Z,
        "P": fields_lo.P - fields_hi.P,
    }

    if match_gradients:
        if lo_theta is None or hi_theta is None:
            raise ValueError("lo_theta and hi_theta are required when match_gradients=True")
        out.update(
            {
                "dTheta_U_R": _grad(fields_lo.U_R, lo_theta) - _grad(fields_hi.U_R, hi_theta),
                "dTheta_U_Theta": _grad(fields_lo.U_Theta, lo_theta) - _grad(fields_hi.U_Theta, hi_theta),
                "dTheta_U_Z": _grad(fields_lo.U_Z, lo_theta) - _grad(fields_hi.U_Z, hi_theta),
                "dTheta_P": _grad(fields_lo.P, lo_theta) - _grad(fields_hi.P, hi_theta),
            }
        )

    return out


def outlet_flow_residual(U_Z_outlet: torch.Tensor, area_weight: torch.Tensor, q_target_nd: float) -> torch.Tensor:
    """Residual for outlet volumetric-flow constraint in non-dimensional form.

    Enforces sum(U_Z * w) = q_target_nd on sampled outlet points.
    """

    q_pred = torch.sum(U_Z_outlet * area_weight)
    q_tar = torch.as_tensor(q_target_nd, dtype=U_Z_outlet.dtype, device=U_Z_outlet.device)
    return q_pred - q_tar
