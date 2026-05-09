from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import NeuralOperators
from NeuralOperators import CFNO2d_small, FNO2d_small, HF_CFNO2d_small, seed_everything


@dataclass(frozen=True)
class AnnularCouetteConfig:
    n: int = 64
    rh: float = 2.0
    rs: float = 4.0
    mu: float = 1.0
    rho: float = 1.0
    omega_out: float = 1.0
    n_blade: int = 1
    modes: int = 12
    width: int = 32
    depth: int = 4
    epochs: int = 1200
    lr: float = 2e-3
    weight_decay: float = 1e-5
    print_interval: int = 200
    seed: int = 10492


class AnnularCouetteTheory:
    def __init__(self, config: AnnularCouetteConfig, device: torch.device):
        self.config = config
        self.device = device
        self.delta_r = config.rs - config.rh
        self.theta0 = 2.0 * np.pi / config.n_blade
        self.u_omega = config.rs * config.omega_out
        self.p0 = 0.5 * config.rho * self.u_omega ** 2
        self.re_omega = self.u_omega * self.delta_r / (config.mu / config.rho)
        self.eu_omega = self.p0 / (config.rho * self.u_omega ** 2)
        self._build()

    def _build(self) -> None:
        n = self.config.n
        r_unit = torch.linspace(0.0, 1.0, n, device=self.device)
        theta_unit = torch.linspace(0.0, 1.0, n, device=self.device)
        rr_unit, tt_unit = torch.meshgrid(r_unit, theta_unit, indexing="ij")
        r_phys = self.config.rh + rr_unit * self.delta_r
        r_hat = r_phys / self.delta_r

        a = self.config.omega_out * self.config.rs ** 2 / (self.config.rs ** 2 - self.config.rh ** 2)
        b = -a * self.config.rh ** 2
        u_theta = a * r_phys + b / torch.clamp(r_phys, min=1e-12)
        ut = u_theta / self.u_omega
        ur = torch.zeros_like(ut)

        d_r = 1.0 / (n - 1)
        d_p_d_r = (1.0 / self.eu_omega) * ut[:, 0] ** 2 / torch.clamp(r_hat[:, 0], min=1e-12)
        p_1d = torch.zeros_like(d_p_d_r)
        for i in range(1, n):
            p_1d[i] = p_1d[i - 1] + 0.5 * (d_p_d_r[i - 1] + d_p_d_r[i]) * d_r
        pressure = p_1d.view(-1, 1).expand(-1, n)

        radial_shape = rr_unit * (1.0 - rr_unit)
        inner_wall = torch.zeros_like(rr_unit)
        outer_wall = torch.zeros_like(rr_unit)
        inner_wall[0, :] = 1.0
        outer_wall[-1, :] = 1.0

        self.r_unit = rr_unit
        self.theta_unit = tt_unit
        self.r_phys = r_phys
        self.theta_phys = tt_unit * self.theta0
        self.r_hat = r_hat
        self.radial_shape = radial_shape
        self.input = torch.stack(
            [
                rr_unit,
                tt_unit,
                r_hat,
                radial_shape,
                inner_wall,
                outer_wall,
            ],
            dim=0,
        ).unsqueeze(0)
        self.target = torch.stack([ur, ut, pressure], dim=0).unsqueeze(0)


class HardConstrainedAnnularSurrogate(nn.Module):
    def __init__(self, core: nn.Module, r_unit: torch.Tensor):
        super().__init__()
        self.core = core
        self.register_buffer("r_unit", r_unit.view(1, 1, *r_unit.shape))
        self.register_buffer("radial_shape", (r_unit * (1.0 - r_unit)).view(1, 1, *r_unit.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.core(x)
        ur = self.radial_shape * raw[:, 0:1]
        ut = self.r_unit + self.radial_shape * raw[:, 1:2]
        p = raw[:, 2:3] - raw[:, 2:3, 0:1, :]
        return torch.cat([ur, ut, p], dim=1)


def make_model(name: str, config: AnnularCouetteConfig, theory: AnnularCouetteTheory) -> nn.Module:
    common = dict(
        modes=config.modes,
        width=config.width,
        depth=config.depth,
        input_features=theory.input.shape[1],
        output_features=3,
    )
    if name == "FNO":
        core = FNO2d_small(**common)
    elif name == "CFNO":
        core = CFNO2d_small(cheb_modes=(config.modes, config.modes), **common)
    elif name == "HF_CFNO":
        core = HF_CFNO2d_small(
            cheb_modes=(config.modes, config.modes),
            high_modes=max(4, config.modes),
            fourier_feature_bands=(1, 2, 4, 8),
            high_gate_init=-2.0,
            use_local_highpass=True,
            **common,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
    return HardConstrainedAnnularSurrogate(core, theory.r_unit)


def channel_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    names = ["UR", "UT", "P"]
    metrics: dict[str, float] = {}
    diff = pred - target
    for index, name in enumerate(names):
        item = diff[:, index]
        metrics[f"{name}_rmse"] = float(torch.sqrt(torch.mean(item ** 2)).detach().cpu().item())
        metrics[f"{name}_max_abs"] = float(torch.max(torch.abs(item)).detach().cpu().item())
    metrics["total_mse"] = float(torch.mean(diff ** 2).detach().cpu().item())
    return metrics


def train_model(
    name: str,
    model_factory: Callable[[], nn.Module],
    theory: AnnularCouetteTheory,
    config: AnnularCouetteConfig,
    output_dir: Path,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    seed_everything(config.seed)
    model = model_factory().to(theory.device)
    x = theory.input.to(theory.device)
    target = theory.target.to(theory.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.05)
    channel_weight = torch.tensor([1.0, 1.0, 1.0], device=theory.device).view(1, 3, 1, 1)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        data_loss = torch.mean(channel_weight * (pred - target) ** 2)
        theta_variation = torch.mean((pred - pred.mean(dim=3, keepdim=True)) ** 2)
        loss = data_loss + 1e-5 * theta_variation
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % config.print_interval == 0 or epoch == config.epochs:
            with torch.no_grad():
                metrics = channel_metrics(model(x), target)
            record = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu().item()),
                "data_loss": float(data_loss.detach().cpu().item()),
                "theta_variation": float(theta_variation.detach().cpu().item()),
                **metrics,
            }
            history.append(record)
            print(
                f"{name:7s} epoch {epoch:04d} | "
                f"loss={record['loss']:.3e} | "
                f"UT_rmse={record['UT_rmse']:.3e} | "
                f"P_rmse={record['P_rmse']:.3e}"
            )

    model.eval()
    with torch.no_grad():
        final_metrics = channel_metrics(model(x), target)
    torch.save(
        {
            "model_name": name,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "metrics": final_metrics,
            "history": history,
        },
        output_dir / f"annular_coette_{name.lower()}_surrogate.pt",
    )
    return model, history, final_metrics


def to_numpy(field: torch.Tensor) -> np.ndarray:
    return field.detach().cpu().numpy()


def annular_edge_grid(theory: AnnularCouetteTheory) -> tuple[np.ndarray, np.ndarray]:
    r_centers = to_numpy(theory.r_phys[:, 0])
    n_r = r_centers.shape[0]
    r_edges = np.empty(n_r + 1, dtype=np.float64)
    r_edges[0] = r_centers[0]
    r_edges[-1] = r_centers[-1]
    r_edges[1:-1] = 0.5 * (r_centers[:-1] + r_centers[1:])

    # The last theta node duplicates the first one for periodicity.  Use
    # n_theta - 1 physical cells so the annulus closes cleanly at 2*pi.
    theta_edges = np.linspace(0.0, float(theory.theta_phys.max().item()), theory.theta_phys.shape[1])
    rr_edges, tt_edges = np.meshgrid(r_edges, theta_edges, indexing="ij")
    x_edges = rr_edges * np.cos(tt_edges)
    y_edges = rr_edges * np.sin(tt_edges)
    return x_edges, y_edges


def plot_profiles(
    theory: AnnularCouetteTheory,
    predictions: dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    target = theory.target[0]
    r = to_numpy(theory.r_phys[:, 0])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    specs = [
        (0, "UR", "Dimensionless radial velocity"),
        (1, "UT", "Dimensionless tangential velocity"),
        (2, "P", "Dimensionless pressure"),
    ]
    for ax, (channel, label, ylabel) in zip(axes[0], specs):
        ax.plot(r, to_numpy(target[channel, :, 0]), "k-", linewidth=2.4, label="Theory")
        for name, pred in predictions.items():
            ax.plot(r, to_numpy(pred[0, channel, :, 0]), "--", linewidth=1.8, label=name)
        ax.set_xlabel("Radius r")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
    axes[0, 1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "annular_coette_profile_comparison.png", dpi=220)
    plt.close(fig)


def plot_field_comparison(
    theory: AnnularCouetteTheory,
    predictions: dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    target = theory.target[0]
    columns = [("Theory", theory.target)] + list(predictions.items())
    x_edges, y_edges = annular_edge_grid(theory)

    fig, axes = plt.subplots(2, len(columns), figsize=(4.2 * len(columns), 8.2), squeeze=False)
    for col, (name, field) in enumerate(columns):
        item = field[0]
        for row, (channel, label) in enumerate([(1, "UT"), (2, "P")]):
            ax = axes[row, col]
            data = to_numpy(item[channel])[:, :-1]
            im = ax.pcolormesh(x_edges, y_edges, data, shading="flat", cmap="cividis")
            ax.set_aspect("equal")
            ax.set_title(f"{name} {label}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "annular_coette_field_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(predictions), figsize=(4.2 * len(predictions), 8.2), squeeze=False)
    for col, (name, pred) in enumerate(predictions.items()):
        error = torch.abs(pred[0] - target)
        for row, (channel, label) in enumerate([(1, "UT error"), (2, "P error")]):
            ax = axes[row, col]
            im = ax.pcolormesh(x_edges, y_edges, to_numpy(error[channel])[:, :-1], shading="flat", cmap="magma")
            ax.set_aspect("equal")
            ax.set_title(f"{name} {label}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "annular_coette_error_maps.png", dpi=220)
    plt.close(fig)


def plot_history(histories: dict[str, list[dict[str, float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    keys = [("loss", "Training loss"), ("UT_rmse", "UT RMSE"), ("P_rmse", "P RMSE")]
    for ax, (key, title) in zip(axes[0], keys):
        for name, history in histories.items():
            epochs = [item["epoch"] for item in history]
            values = [max(item[key], 1e-14) for item in history]
            ax.plot(epochs, values, marker="o", linewidth=1.6, label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "annular_coette_training_history.png", dpi=220)
    plt.close(fig)


def main() -> None:
    config = AnnularCouetteConfig()
    output_dir = Path("theoretical_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theory = AnnularCouetteTheory(config, device=device)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir.resolve()}")

    histories: dict[str, list[dict[str, float]]] = {}
    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, torch.Tensor] = {}

    for name in ["FNO", "CFNO", "HF_CFNO"]:
        factory = lambda model_name=name: make_model(model_name, config, theory)
        model, history, final_metrics = train_model(name, factory, theory, config, output_dir)
        histories[name] = history
        metrics[name] = final_metrics
        with torch.no_grad():
            predictions[name] = model(theory.input.to(device)).detach().cpu()

    theory.target = theory.target.detach().cpu()
    theory.input = theory.input.detach().cpu()
    theory.r_unit = theory.r_unit.detach().cpu()
    theory.r_phys = theory.r_phys.detach().cpu()
    theory.theta_phys = theory.theta_phys.detach().cpu()

    plot_profiles(theory, predictions, output_dir)
    plot_field_comparison(theory, predictions, output_dir)
    plot_history(histories, output_dir)

    result = {
        "config": asdict(config),
        "device": str(device),
        "metrics": metrics,
        "outputs": {
            "profiles": "annular_coette_profile_comparison.png",
            "fields": "annular_coette_field_comparison.png",
            "errors": "annular_coette_error_maps.png",
            "history": "annular_coette_training_history.png",
        },
    }
    (output_dir / "annular_coette_surrogate_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
