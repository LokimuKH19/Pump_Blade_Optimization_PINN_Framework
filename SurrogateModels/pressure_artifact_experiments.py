from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from SurrogateModeling import SurrogateModeling, find_first_blade_params, seed_everything
from SurrogateModelingUtils import expand_scalar, neighbor_plus


def weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.clamp(torch.sum(weight), min=1e-12)


def pressure_diagnostics(trainer: SurrogateModeling) -> dict[str, float | torch.Tensor]:
    trainer.model.eval()
    if trainer.ibm_mask_controller is not None:
        trainer.ibm_mask_controller.eval()
    batch = next(iter(trainer.train_loader))
    batch = trainer._to_device(batch)
    batch = trainer._prepare_runtime_batch(batch)

    with torch.no_grad():
        pred = trainer.model(batch["x"], batch["phi"], batch["solid_ut"])
        pred = trainer._apply_kkt_projection(pred, batch)
        _, log_phys = trainer.physics_loss(pred, batch)

    p = pred["P"]
    mask = trainer.physics_loss.build_pde_mask(batch["phi"])
    p_mean = weighted_mean(p, mask)
    p0 = p - p_mean
    p_energy = weighted_mean(p0 ** 2, mask)
    hp = trainer.physics_loss.pressure_highpass(p)
    hp_energy = weighted_mean(hp ** 2, mask)

    dR = expand_scalar(batch["dR"])
    dTheta = expand_scalar(batch["dTheta"])
    dZ = expand_scalar(batch["dZ"])
    central_grad_energy = weighted_mean(
        trainer.physics_loss.d1(p, dim=1, spacing=dR, periodic=False) ** 2
        + trainer.physics_loss.d1(p, dim=2, spacing=dTheta, periodic=True, duplicate_endpoint=True) ** 2
        + trainer.physics_loss.d1(p, dim=3, spacing=dZ, periodic=False) ** 2,
        mask,
    )
    forward_grad_energy = weighted_mean(
        ((neighbor_plus(p, dim=1, periodic=False) - p) / dR) ** 2
        + ((neighbor_plus(p, dim=2, periodic=True) - p) / dTheta) ** 2
        + ((neighbor_plus(p, dim=3, periodic=False) - p) / dZ) ** 2,
        mask,
    )

    device = p.device
    dtype = p.dtype
    r_sign = ((torch.arange(p.shape[1], device=device) % 2) * 2 - 1).to(dtype).view(1, -1, 1, 1)
    t_sign = ((torch.arange(p.shape[2], device=device) % 2) * 2 - 1).to(dtype).view(1, 1, -1, 1)
    z_sign = ((torch.arange(p.shape[3], device=device) % 2) * 2 - 1).to(dtype).view(1, 1, 1, -1)

    def odd_even_amplitude(pattern: torch.Tensor) -> torch.Tensor:
        amplitude = weighted_mean(p0 * pattern, mask)
        return (amplitude ** 2) / torch.clamp(p_energy, min=1e-12)

    return {
        "loss_phys": float(log_phys["loss_phys"].detach().cpu().item()),
        "loss_qv": float(log_phys["loss_qv"].detach().cpu().item()),
        "loss_p_highpass": float(log_phys["loss_p_highpass"].detach().cpu().item()),
        "loss_p_highpass_ratio": float(log_phys["loss_p_highpass_ratio"].detach().cpu().item()),
        "pressure_rms": float(torch.sqrt(torch.clamp(p_energy, min=0.0)).detach().cpu().item()),
        "pressure_highpass_ratio": float((hp_energy / torch.clamp(p_energy, min=1e-12)).detach().cpu().item()),
        "forward_over_central_pressure_grad": float(
            (forward_grad_energy / torch.clamp(central_grad_energy, min=1e-12)).detach().cpu().item()
        ),
        "odd_even_r": float(odd_even_amplitude(r_sign).detach().cpu().item()),
        "odd_even_theta": float(odd_even_amplitude(t_sign).detach().cpu().item()),
        "odd_even_z": float(odd_even_amplitude(z_sign).detach().cpu().item()),
        "odd_even_theta_z": float(odd_even_amplitude(t_sign * z_sign).detach().cpu().item()),
        "pressure_slice": p[0, p.shape[1] // 2].detach().cpu(),
    }


def build_trainer(blade_params: Path, variant: dict) -> SurrogateModeling:
    kwargs = dict(
        blade_params=blade_params,
        n=16,
        batch_size=1,
        mu=0.006,
        rho=10650.0,
        omega=-210.0 * 2 * torch.pi / 60.0,
        qv=0.16,
        lr=2e-3,
        modes=4,
        width=8,
        depth=2,
        z_padding=2,
        learn_ibm_params=False,
        device="cpu",
    )
    kwargs.update(variant)
    with contextlib.redirect_stdout(io.StringIO()):
        return SurrogateModeling.build_pure_physics_debug_trainer(**kwargs)


def main() -> None:
    blade_params = find_first_blade_params("../BladeOptimizerLFR/CQ_20260327_232449_RealExp_Calc")
    if blade_params is None:
        raise FileNotFoundError("No blade_params.json found for pressure artifact experiments.")

    variants = [
        (
            "cfno_reference",
            dict(operator_variant="cfno"),
        ),
        (
            "hf_aggressive",
            dict(operator_variant="hf_cfno", high_modes=8, fourier_feature_bands=(1, 2, 4, 8)),
        ),
        (
            "hf_cooled",
            dict(
                operator_variant="hf_cfno",
                high_modes=3,
                fourier_feature_bands=(1, 2),
                hf_high_gate_init=-3.0,
                hf_use_local_highpass=False,
            ),
        ),
        (
            "hf_pressure_smooth_05",
            dict(
                operator_variant="hf_cfno",
                high_modes=8,
                fourier_feature_bands=(1, 2, 4, 8),
                pressure_smoothing=0.5,
            ),
        ),
        (
            "hf_pressure_penalty_1e8",
            dict(
                operator_variant="hf_cfno",
                high_modes=8,
                fourier_feature_bands=(1, 2, 4, 8),
                pressure_highpass_weight=1e8,
            ),
        ),
        (
            "hf_kkt_projection",
            dict(
                operator_variant="hf_cfno",
                high_modes=8,
                fourier_feature_bands=(1, 2, 4, 8),
                use_kkt_projection=True,
                kkt_projection_iters=12,
                kkt_projection_strength=0.35,
            ),
        ),
    ]

    output_dir = Path("surrogate_debug_outputs")
    output_dir.mkdir(exist_ok=True)

    rows = []
    slices: list[tuple[str, torch.Tensor]] = []
    epochs = 60

    for index, (name, variant) in enumerate(variants):
        seed_everything(2026)
        trainer = build_trainer(blade_params, variant)
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.fit(epochs=epochs, print_interval=epochs)
        elapsed = time.perf_counter() - start
        diag = pressure_diagnostics(trainer)
        pressure_slice = diag.pop("pressure_slice")
        first = history[0]
        last = history[-1]
        row = {
            "name": name,
            "epochs": epochs,
            "seconds": round(elapsed, 3),
            "train_total_first": first["train_loss_total"],
            "train_total_last": last["train_loss_total"],
            "loss_drop_ratio": first["train_loss_total"] / max(last["train_loss_total"], 1e-30),
        }
        row.update(diag)
        rows.append(row)
        slices.append((name, pressure_slice))
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for ax, (name, pressure_slice) in zip(axes.reshape(-1), slices):
        im = ax.imshow(pressure_slice.numpy(), origin="lower", aspect="auto", cmap="coolwarm")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Z")
        ax.set_ylabel("Theta")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.reshape(-1)[len(slices) :]:
        ax.axis("off")
    fig.tight_layout()
    figure_path = output_dir / "pressure_artifact_experiments.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    result_path = output_dir / "pressure_artifact_experiments.json"
    result_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(figure_path), "results": str(result_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
