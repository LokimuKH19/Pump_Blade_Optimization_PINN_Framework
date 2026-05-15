from __future__ import annotations

from typing import Mapping

import torch

from KKTProjectionOperators import CylindricalDivergenceKKTProjection, KKTProjectionConfig


def build_kkt_projection(
    *,
    enabled: bool,
    iterations: int,
    strength: float,
    device: torch.device,
) -> CylindricalDivergenceKKTProjection | None:
    if not enabled:
        return None
    return CylindricalDivergenceKKTProjection(
        KKTProjectionConfig(iterations=int(iterations), strength=float(strength))
    ).to(device)


def apply_kkt_projection(
    projection: CylindricalDivergenceKKTProjection | None,
    pred: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if projection is None:
        return dict(pred)
    return projection(pred, batch)
