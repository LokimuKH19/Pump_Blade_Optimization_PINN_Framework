# KKTProjectionOperators.py
# Referenced KKTPINN.pdf
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from SurrogateModelingUtils import (
    d1_periodic_with_overlap,
    expand_scalar,
    neighbor_minus,
    neighbor_plus,
)


@dataclass
class KKTProjectionConfig:
    iterations: int = 24
    strength: float = 0.35
    ridge: float = 1e-6
    tolerance: float = 0.0


class CylindricalDivergenceKKTProjection(nn.Module):
    # Differentiable function-space projection:
    #   u* = u_hat - grad(lambda)
    #   -div(grad(lambda)) = -div(u_hat)
    # This is the PDE analogue of B^T(BB^T)^-1 B in KKT-hPINN.
    def __init__(self, config: KKTProjectionConfig | None = None):
        super().__init__()
        self.config = config or KKTProjectionConfig()

    def d1(
        self,
        x: torch.Tensor,
        dim: int,
        spacing: torch.Tensor,
        *,
        periodic: bool,
        duplicate_endpoint: bool = False,
    ) -> torch.Tensor:
        if periodic and duplicate_endpoint:
            return d1_periodic_with_overlap(x, dim, spacing)
        xp = neighbor_plus(x, dim, periodic)
        xm = neighbor_minus(x, dim, periodic)
        return (xp - xm) / (2.0 * spacing)

    def divergence(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        r_hat = batch["r_hat"]
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])

        div_r = self.d1(r_hat * ur, dim=1, spacing=dR, periodic=False) / torch.clamp(r_hat, min=1e-12)
        div_theta = k_theta * self.d1(ut, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        div_z = Lambda * Ku * self.d1(uz, dim=3, spacing=dZ, periodic=False)
        return div_r + div_theta + div_z

    def gradient(
        self,
        pressure: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_theta = batch["K_theta"]
        dR = expand_scalar(batch["dR"])
        dTheta = expand_scalar(batch["dTheta"])
        dZ = expand_scalar(batch["dZ"])
        Lambda = expand_scalar(batch["Lambda"])
        Ku = expand_scalar(batch["Ku"])

        grad_r = self.d1(pressure, dim=1, spacing=dR, periodic=False)
        grad_theta = k_theta * self.d1(pressure, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True)
        grad_z = Lambda * Ku * self.d1(pressure, dim=3, spacing=dZ, periodic=False)
        return grad_r, grad_theta, grad_z

    def normal_operator(self, pressure: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        grad_r, grad_theta, grad_z = self.gradient(pressure, batch)
        return -self.divergence(grad_r, grad_theta, grad_z, batch) + self.config.ridge * pressure

    def _inner(self, a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.sum(a * b * weight, dim=(1, 2, 3), keepdim=True)

    def solve_multiplier(
        self,
        rhs: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        weight: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.zeros_like(rhs)
        r = rhs - self.normal_operator(p, batch)
        r = r - torch.mean(r, dim=(1, 2, 3), keepdim=True)
        direction = r
        rs_old = self._inner(r, r, weight)

        for _ in range(max(int(self.config.iterations), 1)):
            ap = self.normal_operator(direction, batch)
            denom = torch.clamp(self._inner(direction, ap, weight), min=1e-20)
            alpha = rs_old / denom
            p = p + alpha * direction
            p = p - torch.mean(p, dim=(1, 2, 3), keepdim=True)
            r = r - alpha * ap
            r = r - torch.mean(r, dim=(1, 2, 3), keepdim=True)
            rs_new = self._inner(r, r, weight)
            if self.config.tolerance > 0.0 and bool(torch.max(rs_new).detach().cpu() < self.config.tolerance):
                break
            beta = rs_new / torch.clamp(rs_old, min=1e-20)
            direction = r + beta * direction
            rs_old = rs_new

        return p

    def apply_hard_velocity_constraints(
        self,
        ur: torch.Tensor,
        ut: torch.Tensor,
        uz: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = torch.clamp(batch["phi"], min=0.0, max=1.0)
        solid = 1.0 - phi
        solid_ut = batch["solid_ut"]

        ur = phi * ur
        ut = phi * ut + solid * solid_ut
        uz = phi * uz

        ur = ur.clone()
        ut = ut.clone()
        uz = uz.clone()
        ur[:, 0, :, :] = 0.0
        ur[:, -1, :, :] = 0.0
        uz[:, 0, :, :] = 0.0
        uz[:, -1, :, :] = 0.0
        ut[:, 0, :, :] = solid_ut[:, 0, :, :]
        ut[:, -1, :, :] = solid_ut[:, -1, :, :]
        return ur, ut, uz

    def forward(
        self,
        pred: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ur = pred["UR"]
        ut = pred["UT"]
        uz = pred["UZ"]

        weight = torch.clamp(batch["phi"], min=0.0, max=1.0)
        weight = torch.where(weight > 0.5, weight, torch.zeros_like(weight))

        div_before = self.divergence(ur, ut, uz, batch)
        rhs = -div_before * weight
        multiplier = self.solve_multiplier(rhs, batch, torch.clamp(weight, min=1e-6))
        grad_r, grad_theta, grad_z = self.gradient(multiplier, batch)

        strength = float(self.config.strength)
        ur_projected = ur - strength * grad_r
        ut_projected = ut - strength * grad_theta
        uz_projected = uz - strength * grad_z
        ur_projected, ut_projected, uz_projected = self.apply_hard_velocity_constraints(
            ur_projected,
            ut_projected,
            uz_projected,
            batch,
        )
        div_after = self.divergence(ur_projected, ut_projected, uz_projected, batch)

        out = dict(pred)
        out["UR"] = ur_projected
        out["UT"] = ut_projected
        out["UZ"] = uz_projected
        out["kkt_multiplier"] = multiplier
        out["kkt_divergence_before"] = torch.mean(torch.abs(div_before * weight), dim=(1, 2, 3))
        out["kkt_divergence_after"] = torch.mean(torch.abs(div_after * weight), dim=(1, 2, 3))
        return out
