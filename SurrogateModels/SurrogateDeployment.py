from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from BladeImport import bezier_curve, spline_thickness
from SurrogateModeling import BladeFlowDataset, SurrogateModeling, load_checkpoint_payload


ROOT = Path(__file__).resolve().parent
FIELD_NAMES = ("UR", "UT", "UZ", "P")


@dataclass(frozen=True)
class ModelCatalogEntry:
    path: Path
    name: str
    operator_variant: str
    grid_size: int | None
    training_mode: str
    modified_time: float
    size_bytes: int

    @property
    def label(self) -> str:
        grid = f"n={self.grid_size}" if self.grid_size else "n=?"
        variant = self.operator_variant or "unknown"
        return f"{self.name} · {variant} · {grid}"


@dataclass(frozen=True)
class ShapeVariable:
    name: str
    layer_index: int
    family: str
    value_index: int
    baseline: float
    lower: float
    upper: float
    label: str


@dataclass(frozen=True)
class PhysicalCoordinates:
    r_m: np.ndarray
    theta_rad: np.ndarray
    z_m: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray


@dataclass(frozen=True)
class HydraulicMetrics:
    near_wall_distance_m: float
    near_wall_cell_count: int
    relative_velocity_mean_m_s: float
    relative_velocity_max_m_s: float
    relative_velocity_p95_m_s: float
    relative_velocity_rms_m_s: float
    feature_velocity_m_s: float
    static_pressure_rise_pa: float
    pressure_head_m: float
    velocity_head_m: float
    elevation_head_m: float
    total_head_m: float
    predicted_flow_rate_m3_s: float
    prescribed_flow_rate_m3_s: float
    flow_mismatch_ratio: float
    mass_imbalance_ratio: float
    driving_torque_n_m: float
    signed_fluid_torque_n_m: float
    shaft_input_power_w: float
    hydraulic_output_power_w: float
    hydraulic_efficiency: float
    flow_direction: int
    geometry_mask_iou: float = float("nan")
    geometry_volume_change_ratio: float = float("nan")
    geometry_phi_relative_l2: float = float("nan")
    geometry_trust_label: str = "unassessed"
    wall_speed_convention: str = "checkpoint_solid_ut"
    notes: tuple[str, ...] = ()
    blade_pressure_torque_n_m: float = float("nan")
    angular_momentum_torque_n_m: float = float("nan")
    blade_pressure_face_count: int = 0
    torque_method: str = "blade_surface_pressure"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["notes"] = "；".join(self.notes)
        return record


SURROGATE_PRESSURE_TORQUE_CALIBRATION = 2.2165868881953865


def effective_pump_metrics(
    metrics: HydraulicMetrics,
    *,
    pressure_torque_calibration_factor: float = 1.0,
) -> HydraulicMetrics:
    """Return pump-oriented head/power magnitudes for a signed CFD convention.

    The stored cases use a negative rotation/through-flow convention, so a
    physically positive pump head is represented by a negative signed total
    head.  Keep the raw convention when it is already positive and otherwise
    reverse only the direction-dependent hydraulic quantities.
    """

    oriented = metrics
    if float(metrics.total_head_m) < 0.0:
        oriented = replace(
            metrics,
            total_head_m=-metrics.total_head_m,
            pressure_head_m=-metrics.pressure_head_m,
            velocity_head_m=-metrics.velocity_head_m,
            elevation_head_m=-metrics.elevation_head_m,
            static_pressure_rise_pa=-metrics.static_pressure_rise_pa,
            hydraulic_output_power_w=-metrics.hydraulic_output_power_w,
            hydraulic_efficiency=-metrics.hydraulic_efficiency,
        )
    factor = float(pressure_torque_calibration_factor)
    if abs(factor - 1.0) <= 1e-12:
        return oriented
    return replace(
        oriented,
        blade_pressure_torque_n_m=oriented.blade_pressure_torque_n_m * factor,
        driving_torque_n_m=oriented.driving_torque_n_m * factor,
        shaft_input_power_w=oriented.shaft_input_power_w * factor,
        hydraulic_efficiency=oriented.hydraulic_efficiency / factor,
        notes=tuple(oriented.notes) + (f"叶面压力转矩采用三参考叶型中位比值校准系数 {factor:.6f}。",),
    )


class EffectivePumpMetricsEngine:
    """Adapter exposing positive pump-oriented metrics to the optimizer."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def predict_metrics_many(self, parameter_sets: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[HydraulicMetrics]:
        return [
            effective_pump_metrics(
                item,
                pressure_torque_calibration_factor=SURROGATE_PRESSURE_TORQUE_CALIBRATION,
            )
            for item in self.engine.predict_metrics_many(parameter_sets, **kwargs)
        ]


@dataclass
class DeploymentPrediction:
    case: dict[str, Any]
    blade_parameters: dict[str, Any]
    fields_dimensionless: dict[str, np.ndarray]
    fields_physical: dict[str, np.ndarray]
    coordinates: PhysicalCoordinates
    blade_mask: np.ndarray
    phi: np.ndarray
    wall_distance_m: np.ndarray
    near_wall_mask: np.ndarray
    relative_speed_m_s: np.ndarray
    metrics: HydraulicMetrics
    checkpoint_path: str = ""
    compatibility_migration: str | None = None


@dataclass
class OptimizationCandidate:
    values: tuple[float, ...]
    blade_parameters: dict[str, Any]
    metrics: HydraulicMetrics | None
    objectives: tuple[float, float]
    constraint_violation: float
    generation: int
    error: str | None = None

    @property
    def feasible(self) -> bool:
        return self.error is None and math.isfinite(self.constraint_violation) and self.constraint_violation <= 1e-12


@dataclass
class OptimizationResult:
    variables: list[ShapeVariable]
    candidates: list[OptimizationCandidate]
    pareto_indices: list[int]
    recommended_index: int | None
    target_head_m: float
    feature_metric: str
    total_evaluations: int
    feasible_count: int
    message: str

    @property
    def pareto_candidates(self) -> list[OptimizationCandidate]:
        return [self.candidates[index] for index in self.pareto_indices]

    @property
    def recommended(self) -> OptimizationCandidate | None:
        return None if self.recommended_index is None else self.candidates[self.recommended_index]


def _read_run_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = checkpoint_path.parent / "run_config_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def discover_model_checkpoints(root: str | Path = ROOT / "surrogate_formal") -> list[ModelCatalogEntry]:
    """Discover checkpoint files without loading their large tensor payloads."""

    root = Path(root)
    if not root.exists():
        return []
    preferred = {
        "CFNO-VeryGoodResult-Upwind": 0,
        "HFCFNO-VeryGoodResult-Upwind": 1,
        "500EpochBiHF-CFNO-1504596Paras": 2,
        "500EpochBiHF-FNO-1464930Paras": 3,
    }
    entries: list[ModelCatalogEntry] = []
    for path in root.rglob("surrogate_checkpoint.pt"):
        if path.name.endswith(".tmp"):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        summary = _read_run_summary(path)
        model_config = summary.get("model_config", {}) if isinstance(summary, Mapping) else {}
        data_config = summary.get("data_config", {}) if isinstance(summary, Mapping) else {}
        entries.append(
            ModelCatalogEntry(
                path=path.resolve(),
                name=path.parent.name,
                operator_variant=str(model_config.get("operator_variant", "")),
                grid_size=int(data_config["n"]) if data_config.get("n") is not None else None,
                training_mode=str(summary.get("training_mode", "")) if isinstance(summary, Mapping) else "",
                modified_time=float(stat.st_mtime),
                size_bytes=int(stat.st_size),
            )
        )
    entries.sort(key=lambda item: (preferred.get(item.name, 100), -item.modified_time, item.name.lower()))
    return entries


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read deployment metadata and identify the audited legacy HF format."""

    checkpoint_path = Path(path).resolve()
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model_config = dict(payload.get("model_config", {}))
    variant = str(model_config.get("operator_variant", "unknown"))
    state_keys = set(payload.get("model_state_dict", {}))
    has_legacy_high_z = any(key.endswith(".band_spectral.weights_high_pos") for key in state_keys)
    has_directional_high = any(key.endswith(".band_spectral.weights_high_y_pos") for key in state_keys)
    legacy_hf = variant in {"hf_cfno", "hf_fno"} and has_legacy_high_z and not has_directional_high
    train_cases = [dict(item) for item in payload.get("train_case_summaries", [])]
    return {
        "path": checkpoint_path,
        "checkpoint_version": payload.get("checkpoint_version"),
        "input_mode": payload.get("input_mode"),
        "model_config": model_config,
        "trainer_config": dict(payload.get("trainer_config", {})),
        "train_case_summaries": train_cases,
        "val_case_summaries": [dict(item) for item in payload.get("val_case_summaries", [])],
        "history_length": len(payload.get("history") or []),
        "compatibility": "legacy_hf_zero_fill" if legacy_hf else "native_strict",
    }


def _candidate_reference_paths(raw_path: str | Path, checkpoint_path: Path | None = None) -> list[Path]:
    raw = Path(raw_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([ROOT / raw, ROOT.parent / raw])
        if checkpoint_path is not None:
            candidates.append(checkpoint_path.parent / raw)
    return candidates


def resolve_blade_parameters_path(raw_path: str | Path, checkpoint_path: str | Path | None = None) -> Path:
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).resolve()
    for candidate in _candidate_reference_paths(raw_path, checkpoint):
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

    raw_text = str(raw_path).replace("/", "\\")
    tokens = [token for token in raw_text.split("\\") if token.upper().startswith("CQ_")]
    blade_root = ROOT.parent / "BladeOptimizerLFR"
    for token in reversed(tokens):
        candidate = blade_root / token / "blade_params.json"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve blade parameter file referenced by checkpoint: {raw_path}")


def load_blade_parameters(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("blade_params.json must contain a JSON object.")
    return value


def load_checkpoint_case(
    checkpoint_path: str | Path,
    case_index: int = 0,
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    checkpoint = Path(checkpoint_path).resolve()
    payload = load_checkpoint_payload(checkpoint, map_location="cpu")
    summaries = list(payload.get("train_case_summaries", []))
    if not summaries:
        raise ValueError("The selected checkpoint does not contain a deployable training-case summary.")
    if not 0 <= int(case_index) < len(summaries):
        raise IndexError(f"case_index={case_index} is outside the {len(summaries)} stored cases.")
    case = dict(summaries[int(case_index)])
    raw_blade_path = case.get("blade_params")
    if raw_blade_path is None:
        raise ValueError("The stored work condition does not reference blade_params.json.")
    blade_path = resolve_blade_parameters_path(raw_blade_path, checkpoint)
    blade_parameters = load_blade_parameters(blade_path)
    validate_blade_parameter_contract(blade_parameters)
    case["blade_params"] = copy.deepcopy(blade_parameters)
    return case, blade_path, blade_parameters, dict(payload)


def validate_blade_parameter_contract(
    parameters: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> None:
    layers = parameters.get("blade_layers")
    if not isinstance(layers, list) or len(layers) != 5:
        raise ValueError("Deployment design requires exactly five blade layers.")
    baseline_layers = None if baseline is None else baseline.get("blade_layers")
    if baseline is not None and (not isinstance(baseline_layers, list) or len(baseline_layers) != 5):
        raise ValueError("Baseline blade parameters must contain the same five layers.")

    for layer_index, layer in enumerate(layers):
        camber = np.asarray(layer.get("camber_ctrl"), dtype=float)
        thickness = layer.get("thickness_knots", {})
        knots_x = np.asarray(thickness.get("x"), dtype=float)
        knots_t = np.asarray(thickness.get("t"), dtype=float)
        if camber.ndim != 1 or camber.size < 4:
            raise ValueError(f"Layer {layer_index + 1}: camber_ctrl must contain at least four values.")
        if knots_x.ndim != 1 or knots_t.ndim != 1 or knots_x.size != knots_t.size or knots_x.size < 3:
            raise ValueError(f"Layer {layer_index + 1}: thickness x/t arrays must have the same length.")
        if not np.all(np.isfinite(camber)) or not np.all(np.isfinite(knots_x)) or not np.all(np.isfinite(knots_t)):
            raise ValueError(f"Layer {layer_index + 1}: shape parameters must be finite.")
        if not np.all(np.diff(knots_x) > 0.0):
            raise ValueError(f"Layer {layer_index + 1}: thickness knot x positions must be strictly increasing.")
        if np.any(knots_t < 0.0):
            raise ValueError(f"Layer {layer_index + 1}: thickness parameters cannot be negative.")
        h_max = float(layer.get("h_max", float("nan")))
        t_max = float(layer.get("t_max", float("nan")))
        if not math.isfinite(h_max) or h_max <= 0.0:
            raise ValueError(f"Layer {layer_index + 1}: h_max must be finite and positive.")
        if not math.isfinite(t_max) or t_max <= 0.0:
            raise ValueError(f"Layer {layer_index + 1}: t_max must be finite and positive.")

        if baseline_layers is not None:
            base_layer = baseline_layers[layer_index]
            base_camber = np.asarray(base_layer["camber_ctrl"], dtype=float)
            base_x = np.asarray(base_layer["thickness_knots"]["x"], dtype=float)
            base_t = np.asarray(base_layer["thickness_knots"]["t"], dtype=float)
            if camber.size != base_camber.size or knots_x.size != base_x.size:
                raise ValueError(f"Layer {layer_index + 1}: array lengths are frozen by the checkpoint geometry.")
            if not np.array_equal(camber[[0, -1]], base_camber[[0, -1]]):
                raise ValueError(f"Layer {layer_index + 1}: camber leading/trailing endpoints are fixed.")
            if not np.array_equal(knots_t[[0, -1]], base_t[[0, -1]]):
                raise ValueError(f"Layer {layer_index + 1}: thickness leading/trailing endpoints are fixed.")
            if not np.array_equal(knots_x, base_x):
                raise ValueError(f"Layer {layer_index + 1}: thickness knot positions are fixed.")
            for key in set(thickness) | set(base_layer["thickness_knots"]):
                if key == "t":
                    continue
                if thickness.get(key) != base_layer["thickness_knots"].get(key):
                    raise ValueError(f"Layer {layer_index + 1}: thickness field {key} is locked.")

    if baseline is not None:
        for key in set(parameters) | set(baseline):
            if key == "blade_layers":
                continue
            if parameters.get(key) != baseline.get(key):
                raise ValueError(f"Top-level blade field {key} is locked to the stored work condition.")
        for index, layer in enumerate(layers):
            base_layer = baseline_layers[index]
            for key in set(layer) | set(base_layer):
                if key in {"camber_ctrl", "thickness_knots", "h_max", "t_max"}:
                    continue
                if layer.get(key) != base_layer.get(key):
                    raise ValueError(f"Layer {index + 1}: {key} is locked and cannot be changed.")
    _validate_blade_surface_clearance(parameters)


def _validate_blade_surface_clearance(parameters: Mapping[str, Any]) -> None:
    """Reject coupled designs whose two sides cross or leave the passage."""

    layers = parameters["blade_layers"]
    global_parameters = parameters["global_parameters"]
    passage_height = float(global_parameters["passage_height_H1"])
    blade_height = float(global_parameters["blade_height_H"])
    passage_z0 = float(global_parameters.get("z0", 0.0))
    blade_z0 = passage_z0 + 0.5 * (passage_height - blade_height)
    passage_z1 = passage_z0 + passage_height
    layer_s = np.linspace(0.0, 1.0, len(layers))
    sample_s = np.linspace(0.0, 1.0, 33)
    sample_x = np.linspace(0.0, 1.0, 129)
    h_interp = PchipInterpolator(layer_s, [float(layer["h_max"]) for layer in layers])
    t_interp = PchipInterpolator(layer_s, [float(layer["t_max"]) for layer in layers])
    camber_interps = [
        PchipInterpolator(layer_s, [float(layer["camber_ctrl"][index]) for layer in layers])
        for index in range(len(layers[0]["camber_ctrl"]))
    ]
    thickness_interps = [
        PchipInterpolator(layer_s, [float(layer["thickness_knots"]["t"][index]) for layer in layers])
        for index in range(len(layers[0]["thickness_knots"]["t"]))
    ]
    knots_x = np.asarray(layers[0]["thickness_knots"]["x"], dtype=float)
    tolerance = 1e-9 * max(1.0, passage_height)
    for span in sample_s:
        h_max = float(h_interp(span))
        t_max = float(t_interp(span))
        if h_max <= 0.0 or t_max <= 0.0:
            raise ValueError("Spanwise h_max and t_max must stay positive without crossing zero.")
        camber = np.asarray([interpolator(span) for interpolator in camber_interps], dtype=float)
        thickness = np.asarray([interpolator(span) for interpolator in thickness_interps], dtype=float)
        gamma = bezier_curve(sample_x, camber)
        tau = spline_thickness(sample_x, knots_x, thickness)
        center = blade_z0 + sample_x * blade_height - h_max * gamma
        lower = center - t_max * tau
        upper = center + t_max * tau
        if np.any(lower > upper + tolerance):
            raise ValueError("Blade lower and upper surfaces cross after parameter variation.")
        if float(np.min(lower)) < passage_z0 - tolerance or float(np.max(upper)) > passage_z1 + tolerance:
            raise ValueError("Blade surfaces cross the physical axial passage boundary.")


def blade_parameter_rows(parameters: Mapping[str, Any]) -> list[dict[str, float | int]]:
    validate_blade_parameter_contract(parameters)
    rows: list[dict[str, float | int]] = []
    for index, layer in enumerate(parameters["blade_layers"]):
        camber = layer["camber_ctrl"]
        thickness = layer["thickness_knots"]["t"]
        if len(camber) != 4 or len(thickness) != 5:
            raise ValueError("The deployment editor currently expects 4 camber controls and 5 thickness knots per layer.")
        rows.append(
            {
                "layer": index + 1,
                "camber_1": float(camber[1]),
                "camber_2": float(camber[2]),
                "thickness_1": float(thickness[1]),
                "thickness_2": float(thickness[2]),
                "thickness_3": float(thickness[3]),
                "h_max": float(layer["h_max"]),
                "t_max": float(layer["t_max"]),
            }
        )
    return rows


def _relative_bounds(value: float, fraction: float, *, nonnegative: bool = False) -> tuple[float, float]:
    span = max(abs(float(value)) * float(fraction), 1e-8)
    lower, upper = float(value) - span, float(value) + span
    if nonnegative:
        lower = max(0.0, lower)
    return lower, upper


def apply_blade_parameter_rows(
    baseline: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    max_variation_fraction: float,
) -> dict[str, Any]:
    if not 0.0 <= float(max_variation_fraction) <= 0.50:
        raise ValueError("The structure-variation window must be between 0% and 50%.")
    if len(rows) != 5:
        raise ValueError("All five layer rows are required.")
    result = copy.deepcopy(dict(baseline))
    baseline_rows = blade_parameter_rows(baseline)
    by_layer = {int(row["layer"]): row for row in rows}
    if set(by_layer) != {1, 2, 3, 4, 5}:
        raise ValueError("Layer identifiers must be exactly 1 through 5.")

    columns = ("camber_1", "camber_2", "thickness_1", "thickness_2", "thickness_3", "h_max", "t_max")
    for base_row in baseline_rows:
        layer_number = int(base_row["layer"])
        edited = by_layer[layer_number]
        values: dict[str, float] = {}
        for column in columns:
            value = float(edited[column])
            if not math.isfinite(value):
                raise ValueError(f"Layer {layer_number}: {column} must be finite.")
            lower, upper = _relative_bounds(
                float(base_row[column]),
                max_variation_fraction,
                nonnegative=column.startswith("thickness") or column in {"h_max", "t_max"},
            )
            tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"Layer {layer_number}: {column}={value:.6g} is outside "
                    f"the allowed [{lower:.6g}, {upper:.6g}]."
                )
            values[column] = value
        layer = result["blade_layers"][layer_number - 1]
        layer["camber_ctrl"][1] = values["camber_1"]
        layer["camber_ctrl"][2] = values["camber_2"]
        layer["thickness_knots"]["t"][1] = values["thickness_1"]
        layer["thickness_knots"]["t"][2] = values["thickness_2"]
        layer["thickness_knots"]["t"][3] = values["thickness_3"]
        layer["h_max"] = values["h_max"]
        layer["t_max"] = values["t_max"]
    validate_blade_parameter_contract(result, baseline=baseline)
    return result


def build_optimization_variables(
    parameters: Mapping[str, Any],
    variation_fraction: float,
) -> list[ShapeVariable]:
    """Build 25 variables: 15 curve ratios plus per-layer h_max and t_max.

    One camber control and one thickness amplitude per layer are held as gauge
    anchors.  Varying all values by a common factor would be removed by the
    normalizations in ``BladeImport.bezier_curve`` and ``spline_thickness``.
    """

    validate_blade_parameter_contract(parameters)
    variables: list[ShapeVariable] = []
    for layer_index, layer in enumerate(parameters["blade_layers"]):
        definitions = [
            ("camber", 1, float(layer["camber_ctrl"][1]), False, "型线比 C1/C2"),
            ("thickness", 2, float(layer["thickness_knots"]["t"][2]), True, "厚度比 T2/T1"),
            ("thickness", 3, float(layer["thickness_knots"]["t"][3]), True, "厚度比 T3/T1"),
            ("h_max", -1, float(layer["h_max"]), True, "弯度幅值 h_max"),
            ("t_max", -1, float(layer["t_max"]), True, "半厚度幅值 t_max"),
        ]
        for family, value_index, baseline, nonnegative, title in definitions:
            lower, upper = _relative_bounds(baseline, variation_fraction, nonnegative=nonnegative)
            variables.append(
                ShapeVariable(
                    name=f"L{layer_index + 1}_{family}_{value_index}",
                    layer_index=layer_index,
                    family=family,
                    value_index=value_index,
                    baseline=baseline,
                    lower=lower,
                    upper=upper,
                    label=f"第{layer_index + 1}层 {title}",
                )
            )
    return variables


def build_training_envelope_variables(
    baseline: Mapping[str, Any],
    training_folders: Sequence[str | Path],
    *,
    expansion_fraction: float = 0.0,
    constant_fraction: float = 0.0,
) -> list[ShapeVariable]:
    """Build an interpolation-only search box from training blade JSON files.

    All seven editable values per layer are inspected.  Zero-width dimensions
    are intentionally omitted, so the optimizer cannot invent variation that
    does not occur in the supplied training geometries.
    """

    validate_blade_parameter_contract(baseline)
    parameter_sets: list[dict[str, Any]] = []
    for raw_folder in training_folders:
        folder = Path(raw_folder)
        if not folder.is_absolute():
            folder = (ROOT / folder).resolve()
        blade_path = folder if folder.name.lower() == "blade_params.json" else folder / "blade_params.json"
        if not blade_path.exists():
            raise FileNotFoundError(f"Training blade JSON does not exist: {blade_path}")
        parameters = load_blade_parameters(blade_path)
        validate_blade_parameter_contract(parameters)
        if parameters.get("global_parameters") != baseline.get("global_parameters"):
            raise ValueError(f"Training geometry has incompatible global parameters: {blade_path}")
        parameter_sets.append(parameters)
    if not parameter_sets:
        raise ValueError("At least one training folder is required for a JSON-envelope search.")

    definitions = (
        ("camber", 1, "camber C1"),
        ("camber", 2, "camber C2"),
        ("thickness", 1, "thickness T1"),
        ("thickness", 2, "thickness T2"),
        ("thickness", 3, "thickness T3"),
        ("h_max", -1, "h_max"),
        ("t_max", -1, "t_max"),
    )

    def value_at(parameters: Mapping[str, Any], layer_index: int, family: str, value_index: int) -> float:
        layer = parameters["blade_layers"][layer_index]
        if family == "camber":
            return float(layer["camber_ctrl"][value_index])
        if family == "thickness":
            return float(layer["thickness_knots"]["t"][value_index])
        return float(layer[family])

    variables: list[ShapeVariable] = []
    for layer_index in range(5):
        for family, value_index, title in definitions:
            samples = [value_at(parameters, layer_index, family, value_index) for parameters in parameter_sets]
            lower, upper = float(min(samples)), float(max(samples))
            baseline_value = value_at(baseline, layer_index, family, value_index)
            tolerance = 1e-12 * max(1.0, abs(lower), abs(upper))
            if upper - lower <= tolerance:
                if constant_fraction <= 0.0:
                    continue
                lower, upper = _relative_bounds(
                    baseline_value,
                    constant_fraction,
                    nonnegative=family in {"thickness", "h_max", "t_max"},
                )
            elif expansion_fraction > 0.0:
                padding = (upper - lower) * float(expansion_fraction)
                lower -= padding
                upper += padding
                if family in {"thickness", "h_max", "t_max"}:
                    lower = max(0.0, lower)
            if baseline_value < lower - tolerance or baseline_value > upper + tolerance:
                raise ValueError(f"Baseline value is outside the training envelope for layer {layer_index + 1} {title}.")
            variables.append(
                ShapeVariable(
                    name=f"L{layer_index + 1}_{family}_{value_index}",
                    layer_index=layer_index,
                    family=family,
                    value_index=value_index,
                    baseline=baseline_value,
                    lower=lower,
                    upper=upper,
                    label=f"第{layer_index + 1}层 {title}",
                )
            )
    if not variables:
        raise ValueError("The training JSON envelope has no non-zero-width design variable.")
    return variables


def checkpoint_training_folders(checkpoint_path: str | Path) -> list[Path]:
    """Read the ordered (and possibly duplicated) training-folder list."""

    checkpoint = Path(checkpoint_path).resolve()
    summary_candidates = [checkpoint.parent / "run_config_summary.json"]
    try:
        payload = load_checkpoint_payload(checkpoint, map_location="cpu")
        metadata = payload.get("extra_metadata") if isinstance(payload, Mapping) else None
        if isinstance(metadata, Mapping) and metadata.get("run_summary_path"):
            raw_summary = Path(str(metadata["run_summary_path"]))
            summary_candidates.insert(0, raw_summary if raw_summary.is_absolute() else (ROOT / raw_summary).resolve())
    except RuntimeError:
        pass
    for summary_path in summary_candidates:
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        folders = payload.get("simulation_folders") or []
        resolved: list[Path] = []
        for raw_folder in folders:
            folder = Path(str(raw_folder))
            resolved.append(folder.resolve() if folder.is_absolute() else (ROOT / folder).resolve())
        if resolved:
            return resolved
    return []


def apply_shape_vector(
    baseline: Mapping[str, Any],
    variables: Sequence[ShapeVariable],
    values: Sequence[float],
) -> dict[str, Any]:
    if len(variables) != len(values):
        raise ValueError("Shape vector length does not match its variable definition.")
    result = copy.deepcopy(dict(baseline))
    for variable, raw_value in zip(variables, values):
        value = float(raw_value)
        if not math.isfinite(value) or value < variable.lower - 1e-12 or value > variable.upper + 1e-12:
            raise ValueError(f"{variable.label} is outside its local-generalization bound.")
        layer = result["blade_layers"][variable.layer_index]
        if variable.family == "camber":
            layer["camber_ctrl"][variable.value_index] = value
        elif variable.family == "thickness":
            layer["thickness_knots"]["t"][variable.value_index] = value
        elif variable.family in {"h_max", "t_max"}:
            layer[variable.family] = value
        else:
            raise ValueError(f"Unsupported shape family: {variable.family}")
    validate_blade_parameter_contract(result, baseline=baseline)
    return result


def shape_change_summary(parameters: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    current = blade_parameter_rows(parameters)
    reference = blade_parameter_rows(baseline)
    changes: list[float] = []
    for row, base_row in zip(current, reference):
        for key in ("camber_1", "camber_2", "thickness_1", "thickness_2", "thickness_3", "h_max", "t_max"):
            denominator = max(abs(float(base_row[key])), 1e-12)
            changes.append((float(row[key]) - float(base_row[key])) / denominator)
    values = np.asarray(changes, dtype=float)
    return {
        "max_absolute_fraction": float(np.max(np.abs(values))),
        "rms_fraction": float(np.sqrt(np.mean(values**2))),
    }


def physical_coordinates(case: Mapping[str, Any]) -> PhysicalCoordinates:
    n = int(case["n"])
    r = np.linspace(float(case["rh"]), float(case["rs"]), n, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi / int(case["n_blade"]), n, dtype=np.float64)
    z = np.linspace(float(case.get("z0", 0.0)), float(case.get("z0", 0.0)) + float(case["h"]), n, dtype=np.float64)
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    return PhysicalCoordinates(r_m=r, theta_rad=theta, z_m=z, x_m=rr * np.cos(tt), y_m=rr * np.sin(tt))


def apply_immersed_boundary_hard_constraint(
    fields_dimensionless: Mapping[str, np.ndarray],
    blade_mask: np.ndarray,
    solid_ut_dimensionless: np.ndarray,
) -> dict[str, np.ndarray]:
    """Enforce the binary IBM constraint exactly on the deployed velocity fields.

    The learned ``phi`` projection remains part of the neural operator.  This
    final binary projection uses BladeImport's authoritative solid mask so that
    floating-point/adaptive-phi changes cannot leave residual velocity in the
    blade: UR=UZ=0 and UT=solid_ut in the solid region.  Pressure is retained
    internally for numerics and is removed from fluid-only products below.
    """

    mask = np.asarray(blade_mask, dtype=bool)
    solid_ut = np.asarray(solid_ut_dimensionless)
    if solid_ut.shape != mask.shape:
        raise ValueError("solid_ut and blade_mask must share the same grid shape.")
    constrained: dict[str, np.ndarray] = {}
    for name in FIELD_NAMES:
        values = np.array(fields_dimensionless[name], copy=True)
        if values.shape != mask.shape:
            raise ValueError(f"{name} and blade_mask must share the same grid shape.")
        constrained[name] = values
    constrained["UR"][mask] = 0.0
    constrained["UZ"][mask] = 0.0
    constrained["UT"][mask] = solid_ut[mask]
    return constrained


def fluid_only_field(field: np.ndarray, blade_mask: np.ndarray) -> np.ndarray:
    """Return a physical fluid-domain field with NaN at every solid IBM cell."""

    values = np.asarray(field)
    mask = np.asarray(blade_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("field and blade_mask must share the same grid shape.")
    result = np.array(values, dtype=np.result_type(values.dtype, np.float32), copy=True)
    result[mask] = np.nan
    return result


def _periodic_near_wall_distance(
    blade_mask: np.ndarray,
    case: Mapping[str, Any],
) -> np.ndarray:
    """Approximate metre-scale wall distance from a periodic surface voxel cloud."""

    mask = np.asarray(blade_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("Blade mask is empty; near-wall metrics cannot be computed.")
    n_r, n_theta, n_z = mask.shape
    r_plus = np.zeros_like(mask)
    r_minus = np.zeros_like(mask)
    z_plus = np.zeros_like(mask)
    z_minus = np.zeros_like(mask)
    r_plus[:-1] = mask[1:]
    r_minus[1:] = mask[:-1]
    z_plus[:, :, :-1] = mask[:, :, 1:]
    z_minus[:, :, 1:] = mask[:, :, :-1]
    theta_plus = np.roll(mask, -1, axis=1)
    theta_minus = np.roll(mask, 1, axis=1)
    interior = mask & r_plus & r_minus & z_plus & z_minus & theta_plus & theta_minus
    surface = mask & ~interior

    coordinates = physical_coordinates(case)
    rr, tt, zz = np.meshgrid(coordinates.r_m, coordinates.theta_rad, coordinates.z_m, indexing="ij")
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    surface_points = np.column_stack((xx[surface], yy[surface], zz[surface]))
    if surface_points.size == 0:
        raise ValueError("Blade surface extraction produced no points.")

    sector_angle = 2.0 * np.pi / int(case["n_blade"])
    periodic_points = []
    for angle in (-sector_angle, 0.0, sector_angle):
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = surface_points.copy()
        rotated[:, 0] = cosine * surface_points[:, 0] - sine * surface_points[:, 1]
        rotated[:, 1] = sine * surface_points[:, 0] + cosine * surface_points[:, 1]
        periodic_points.append(rotated)
    tree = cKDTree(np.concatenate(periodic_points, axis=0))
    query_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    distance = tree.query(query_points, k=1, workers=-1)[0].reshape(mask.shape)

    # Surface voxels represent cell centres.  Subtract a local half diagonal to
    # approximate the actual solid/fluid face rather than the solid centre.
    dr = (float(case["rs"]) - float(case["rh"])) / max(n_r - 1, 1)
    dtheta = sector_angle / max(n_theta - 1, 1)
    dz = float(case["h"]) / max(n_z - 1, 1)
    half_diagonal = 0.5 * np.sqrt(dr**2 + (rr * dtheta) ** 2 + dz**2)
    distance = np.maximum(distance - half_diagonal, np.finfo(np.float64).eps)
    distance[mask] = 0.0
    return distance


def _trapezoid_weights(length: int) -> np.ndarray:
    weights = np.ones(int(length), dtype=np.float64)
    if length > 1:
        weights[[0, -1]] = 0.5
    return weights


def integrate_blade_pressure_torque(
    pressure_pa: np.ndarray,
    blade_mask: np.ndarray,
    case: Mapping[str, Any],
) -> tuple[float, int]:
    """Integrate pressure torque on the immersed blade surface.

    ``BladeImport`` supplies the analytic blade geometry and its rasterized
    mask.  Every solid/fluid interface in the periodic theta direction is a
    small radial-axial blade face.  Pressure is linearly extrapolated from the
    first two fluid points to the face and integrated as ``r * F_theta``.
    Stair-stepped theta faces also represent the tangential component of the
    curved upper/lower analytic surfaces as the grid is refined.
    """

    pressure = np.asarray(pressure_pa, dtype=np.float64)
    mask = np.asarray(blade_mask, dtype=bool)
    if pressure.shape != mask.shape or pressure.ndim != 3:
        raise ValueError("Pressure and blade mask must share a 3-D grid.")
    n_r, n_theta, n_z = mask.shape
    if min(n_r, n_theta, n_z) < 2:
        return float("nan"), 0
    radii = np.linspace(float(case["rh"]), float(case["rs"]), n_r, dtype=np.float64)
    dr = (float(case["rs"]) - float(case["rh"])) / max(n_r - 1, 1)
    dz = float(case["h"]) / max(n_z - 1, 1)
    radial_weights = _trapezoid_weights(n_r)
    axial_weights = _trapezoid_weights(n_z)
    face_area = dr * dz * radial_weights[:, None] * axial_weights[None, :]
    torque_one_blade = 0.0
    face_count = 0

    for theta_index in range(n_theta):
        next_index = (theta_index + 1) % n_theta
        left_solid = mask[:, theta_index, :]
        right_solid = mask[:, next_index, :]
        interface = left_solid ^ right_solid
        if not np.any(interface):
            continue
        fluid_on_right = left_solid & ~right_solid
        fluid_on_left = ~left_solid & right_solid

        if np.any(fluid_on_right):
            p1 = pressure[:, next_index, :]
            second_index = (next_index + 1) % n_theta
            p2 = pressure[:, second_index, :]
            second_fluid = ~mask[:, second_index, :]
            surface_pressure = np.where(second_fluid, 1.5 * p1 - 0.5 * p2, p1)
            # solid outward normal is +e_theta, so pressure force is -e_theta.
            torque_one_blade += float(
                np.sum((-radii[:, None]) * surface_pressure * face_area * fluid_on_right)
            )
            face_count += int(np.count_nonzero(fluid_on_right))

        if np.any(fluid_on_left):
            p1 = pressure[:, theta_index, :]
            second_index = (theta_index - 1) % n_theta
            p2 = pressure[:, second_index, :]
            second_fluid = ~mask[:, second_index, :]
            surface_pressure = np.where(second_fluid, 1.5 * p1 - 0.5 * p2, p1)
            # solid outward normal is -e_theta, so pressure force is +e_theta.
            torque_one_blade += float(
                np.sum(radii[:, None] * surface_pressure * face_area * fluid_on_left)
            )
            face_count += int(np.count_nonzero(fluid_on_left))

    return float(int(case["n_blade"]) * torque_one_blade), face_count


def compute_hydraulic_metrics(
    fields_physical: Mapping[str, np.ndarray],
    case: Mapping[str, Any],
    blade_mask: np.ndarray,
    solid_ut_dimensionless: np.ndarray,
    *,
    u_omega: float,
    near_wall_distance_m: float,
    wall_speed_convention: str = "signed_omega_r",
) -> tuple[HydraulicMetrics, np.ndarray, np.ndarray, np.ndarray]:
    """Compute near-wall and pump metrics in physical units."""

    if near_wall_distance_m <= 0.0:
        raise ValueError("Near-wall distance d must be positive.")
    arrays = {key: np.asarray(fields_physical[key], dtype=np.float64) for key in FIELD_NAMES}
    shape = arrays["UR"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("UR, UT, UZ, and P must share the same grid shape.")
    mask = np.asarray(blade_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("Blade mask shape does not match the predicted fields.")

    wall_distance = _periodic_near_wall_distance(mask, case)
    fluid = ~mask
    near_mask = fluid & (wall_distance > 0.0) & (wall_distance <= float(near_wall_distance_m))

    solid_ut = np.asarray(solid_ut_dimensionless, dtype=np.float64)
    if solid_ut.shape != shape:
        raise ValueError("solid_ut shape does not match the predicted fields.")
    coordinates = physical_coordinates(case)
    if bool(case.get("absolute_frame", True)) and wall_speed_convention == "signed_omega_r":
        wall_ut = np.broadcast_to(
            float(case["omega"]) * coordinates.r_m[:, None, None],
            shape,
        )
    elif bool(case.get("absolute_frame", True)) and wall_speed_convention == "checkpoint_solid_ut":
        wall_ut = solid_ut * float(u_omega)
    else:
        wall_ut = np.zeros_like(solid_ut)
    relative_speed = np.sqrt(arrays["UR"] ** 2 + (arrays["UT"] - wall_ut) ** 2 + arrays["UZ"] ** 2)
    near_values = relative_speed[near_mask & np.isfinite(relative_speed)]
    if near_values.size == 0:
        mean_speed = max_speed = p95_speed = rms_speed = float("nan")
    else:
        mean_speed = float(np.mean(near_values))
        max_speed = float(np.max(near_values))
        p95_speed = float(np.percentile(near_values, 95.0))
        rms_speed = float(np.sqrt(np.mean(near_values**2)))
    feature_speed = 0.45 * mean_speed + 0.35 * p95_speed + 0.20 * max_speed

    n_r, n_theta, _ = shape
    dr = (coordinates.r_m[-1] - coordinates.r_m[0]) / max(n_r - 1, 1)
    dtheta = (coordinates.theta_rad[-1] - coordinates.theta_rad[0]) / max(n_theta - 1, 1)
    area_weights = (
        coordinates.r_m[:, None]
        * _trapezoid_weights(n_r)[:, None]
        * _trapezoid_weights(n_theta)[None, :]
        * dr
        * dtheta
    )

    def plane_stats(index: int) -> dict[str, float]:
        plane_fluid = fluid[:, :, index]
        weights = area_weights * plane_fluid
        area = float(np.sum(weights))
        if area <= 0.0:
            raise ValueError("Inlet/outlet plane contains no fluid cells.")
        uz = arrays["UZ"][:, :, index]
        ur = arrays["UR"][:, :, index]
        ut = arrays["UT"][:, :, index]
        pressure = arrays["P"][:, :, index]
        q = float(np.sum(uz * weights))
        p_mean = float(np.sum(pressure * weights) / area)
        v2 = ur**2 + ut**2 + uz**2
        flow_weights = np.abs(uz) * weights
        flow_weight_sum = float(np.sum(flow_weights))
        if flow_weight_sum > 1e-12:
            v2_mean = float(np.sum(v2 * flow_weights) / flow_weight_sum)
        else:
            v2_mean = float(np.sum(v2 * weights) / area)
        return {"q": q, "p": p_mean, "v2": v2_mean}

    low = plane_stats(0)
    high = plane_stats(-1)
    average_signed_q = 0.5 * (low["q"] + high["q"])
    prescribed_q = float(case["qv"])
    flow_direction = 1 if average_signed_q > 1e-12 else -1 if average_signed_q < -1e-12 else (1 if prescribed_q >= 0 else -1)
    if flow_direction > 0:
        inlet, outlet = low, high
        inlet_index, outlet_index = 0, -1
        elevation_head = float(case["h"])
    else:
        inlet, outlet = high, low
        inlet_index, outlet_index = -1, 0
        elevation_head = -float(case["h"])

    rho = float(case["rho"])
    gravity = float(case.get("g", 9.8))
    n_blade = int(case["n_blade"])
    delta_p = outlet["p"] - inlet["p"]
    pressure_head = delta_p / (rho * gravity)
    velocity_head = (outlet["v2"] - inlet["v2"]) / (2.0 * gravity)
    total_head = pressure_head + velocity_head + elevation_head

    predicted_q = n_blade * 0.5 * (abs(low["q"]) + abs(high["q"]))
    mass_imbalance = abs(abs(high["q"]) - abs(low["q"])) / max(0.5 * (abs(high["q"]) + abs(low["q"])), 1e-12)
    flow_mismatch = abs(predicted_q - abs(prescribed_q)) / max(abs(prescribed_q), 1e-12)

    def angular_momentum_flux(index: int) -> float:
        weights = area_weights * fluid[:, :, index]
        along_flow_velocity = flow_direction * arrays["UZ"][:, :, index]
        return float(np.sum(coordinates.r_m[:, None] * arrays["UT"][:, :, index] * along_flow_velocity * weights))

    angular_momentum_torque = rho * n_blade * (
        angular_momentum_flux(outlet_index) - angular_momentum_flux(inlet_index)
    )
    pressure_torque, pressure_face_count = integrate_blade_pressure_torque(arrays["P"], mask, case)
    torque_signed = pressure_torque if math.isfinite(pressure_torque) and pressure_face_count > 0 else angular_momentum_torque
    driving_torque = abs(torque_signed)
    shaft_power = abs(driving_torque * float(case["omega"]))
    hydraulic_power = rho * gravity * abs(prescribed_q) * total_head
    efficiency = hydraulic_power / shaft_power if shaft_power > 1e-12 else float("nan")

    notes: list[str] = [
        "驱动转矩采用BladeImport解析叶型生成的浸没边界面，通过叶面邻近压力线性插值后积分；"
        "进出口角动量通量同时保留为对照。"
    ]
    if near_values.size < 32:
        notes.append("d 范围内近壁采样点较少，最大速度对网格较敏感。")
    if mass_imbalance > 0.05:
        notes.append("代理流场进出口质量不平衡超过 5%。")
    if flow_mismatch > 0.10:
        notes.append("代理预测流量与锁定工况流量偏差超过 10%。")
    if not (0.0 <= efficiency <= 1.0):
        notes.append("水力效率不在 0–100% 物理区间，建议用 CFD 复核该设计。")
    if wall_speed_convention == "checkpoint_solid_ut" and float(case.get("omega", 0.0)) < 0.0:
        notes.append("相对速度按 checkpoint 的无符号 solid_ut 壁速约定计算，而非直接使用 omega*r。")
    elif wall_speed_convention == "signed_omega_r":
        notes.append("相对速度按物理有符号 omega*r 叶面速度计算。")

    metrics = HydraulicMetrics(
        near_wall_distance_m=float(near_wall_distance_m),
        near_wall_cell_count=int(near_values.size),
        relative_velocity_mean_m_s=mean_speed,
        relative_velocity_max_m_s=max_speed,
        relative_velocity_p95_m_s=p95_speed,
        relative_velocity_rms_m_s=rms_speed,
        feature_velocity_m_s=float(feature_speed),
        static_pressure_rise_pa=float(delta_p),
        pressure_head_m=float(pressure_head),
        velocity_head_m=float(velocity_head),
        elevation_head_m=float(elevation_head),
        total_head_m=float(total_head),
        predicted_flow_rate_m3_s=float(predicted_q),
        prescribed_flow_rate_m3_s=float(abs(prescribed_q)),
        flow_mismatch_ratio=float(flow_mismatch),
        mass_imbalance_ratio=float(mass_imbalance),
        driving_torque_n_m=float(driving_torque),
        signed_fluid_torque_n_m=float(torque_signed),
        shaft_input_power_w=float(shaft_power),
        hydraulic_output_power_w=float(hydraulic_power),
        hydraulic_efficiency=float(efficiency),
        flow_direction=int(flow_direction),
        wall_speed_convention=wall_speed_convention,
        notes=tuple(notes),
        blade_pressure_torque_n_m=float(pressure_torque),
        angular_momentum_torque_n_m=float(angular_momentum_torque),
        blade_pressure_face_count=int(pressure_face_count),
        torque_method="blade_surface_pressure",
    )
    return metrics, wall_distance.astype(np.float32), near_mask, relative_speed.astype(np.float32)


class DeploymentEngine:
    """Cached checkpoint runtime for locked-condition shape inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        case_index: int = 0,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.case_template, self.blade_path, self.baseline_parameters, self.checkpoint_payload = load_checkpoint_case(
            self.checkpoint_path,
            case_index=case_index,
        )
        resolved_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.device = resolved_device
        self.trainer = SurrogateModeling.from_checkpoint(
            self.checkpoint_path,
            self.case_template,
            device=resolved_device,
            batch_size=1,
            load_optimizer=False,
        )
        baseline_sample = self.trainer.train_dataset[0]
        self.baseline_mask = baseline_sample["blade_mask"].numpy() > 0.5
        self.baseline_phi = baseline_sample["phi"].numpy().astype(np.float32, copy=False)

    @property
    def compatibility_migration(self) -> str | None:
        return self.trainer.checkpoint_compatibility_migration

    def make_case(self, blade_parameters: Mapping[str, Any]) -> dict[str, Any]:
        validate_blade_parameter_contract(blade_parameters, baseline=self.baseline_parameters)
        case = copy.deepcopy(self.case_template)
        case["blade_params"] = copy.deepcopy(dict(blade_parameters))
        return case

    def _predict_many(
        self,
        blade_parameter_sets: Sequence[Mapping[str, Any]],
        *,
        near_wall_distance_m: float,
        batch_size: int,
        metrics_only: bool,
    ) -> list[DeploymentPrediction] | list[HydraulicMetrics]:
        if not blade_parameter_sets:
            return []
        cases = [self.make_case(parameters) for parameters in blade_parameter_sets]
        dataset = BladeFlowDataset(
            cases,
            input_mode=self.trainer.train_dataset.input_mode,
            pressure_reference=self.trainer.pressure_data_reference,
        )
        outputs: list[DeploymentPrediction] | list[HydraulicMetrics] = []
        effective_batch = max(1, int(batch_size))
        self.trainer.model.eval()
        if self.trainer.ibm_mask_controller is not None:
            self.trainer.ibm_mask_controller.eval()

        with torch.inference_mode():
            for start in range(0, len(dataset), effective_batch):
                stop = min(start + effective_batch, len(dataset))
                samples = [dataset[index] for index in range(start, stop)]
                batch = {
                    key: torch.stack([sample[key] for sample in samples], dim=0).to(self.trainer.device)
                    for key in samples[0]
                }
                runtime = self.trainer._prepare_runtime_batch(batch)
                prediction = self.trainer.model(runtime["x"], runtime["phi"], runtime["solid_ut"])
                prediction = self.trainer._apply_kkt_projection(prediction, runtime)

                for local_index, sample in enumerate(samples):
                    pred_dim_raw = {
                        key: prediction[key][local_index].detach().cpu().numpy().astype(np.float32, copy=False)
                        for key in FIELD_NAMES
                    }
                    blade_mask = sample["blade_mask"].numpy() > 0.5
                    solid_ut = sample["solid_ut"].numpy()
                    pred_dim = apply_immersed_boundary_hard_constraint(pred_dim_raw, blade_mask, solid_ut)
                    u_omega = float(runtime["u_omega"][local_index].detach().cpu().item())
                    u_zo = float(runtime["u_zo"][local_index].detach().cpu().item())
                    p0 = float(runtime["P0"][local_index].detach().cpu().item())
                    pred_phy = {
                        "UR": pred_dim["UR"] * u_omega,
                        "UT": pred_dim["UT"] * u_omega,
                        "UZ": pred_dim["UZ"] * u_zo,
                        "P": pred_dim["P"] * p0,
                    }
                    phi = runtime["phi"][local_index].detach().cpu().numpy().astype(np.float32, copy=False)
                    raw_phi = sample["phi"].numpy().astype(np.float32, copy=False)
                    metrics, distance, near_mask, relative_speed = compute_hydraulic_metrics(
                        pred_phy,
                        cases[start + local_index],
                        blade_mask,
                        solid_ut,
                        u_omega=u_omega,
                        near_wall_distance_m=near_wall_distance_m,
                    )
                    intersection = int(np.count_nonzero(blade_mask & self.baseline_mask))
                    union = int(np.count_nonzero(blade_mask | self.baseline_mask))
                    mask_iou = intersection / max(union, 1)
                    base_volume = int(np.count_nonzero(self.baseline_mask))
                    volume_change = abs(int(np.count_nonzero(blade_mask)) - base_volume) / max(base_volume, 1)
                    phi_delta = float(np.linalg.norm((raw_phi - self.baseline_phi).ravel()))
                    phi_scale = max(float(np.linalg.norm(self.baseline_phi.ravel())), 1e-12)
                    phi_relative_l2 = phi_delta / phi_scale
                    trust_label = "已计算"
                    trust_notes = list(metrics.notes)
                    if mask_iou >= 0.999999 and phi_relative_l2 <= 1e-8 and shape_change_summary(
                        blade_parameter_sets[start + local_index], self.baseline_parameters
                    )["max_absolute_fraction"] > 1e-8:
                        trust_notes.append("参数已改变，当前离散网格上的 mask/phi 保持不变。")
                    metrics = replace(
                        metrics,
                        geometry_mask_iou=float(mask_iou),
                        geometry_volume_change_ratio=float(volume_change),
                        geometry_phi_relative_l2=float(phi_relative_l2),
                        geometry_trust_label=trust_label,
                        notes=tuple(trust_notes),
                    )
                    if metrics_only:
                        outputs.append(metrics)
                    else:
                        outputs.append(
                            DeploymentPrediction(
                                case=cases[start + local_index],
                                blade_parameters=copy.deepcopy(dict(blade_parameter_sets[start + local_index])),
                                fields_dimensionless=pred_dim,
                                fields_physical={
                                    key: fluid_only_field(values, blade_mask) for key, values in pred_phy.items()
                                },
                                coordinates=physical_coordinates(cases[start + local_index]),
                                blade_mask=blade_mask,
                                phi=phi,
                                wall_distance_m=distance,
                                near_wall_mask=near_mask,
                                relative_speed_m_s=fluid_only_field(relative_speed, blade_mask),
                                metrics=metrics,
                                checkpoint_path=str(self.checkpoint_path),
                                compatibility_migration=self.compatibility_migration,
                            )
                        )
        return outputs

    def predict(
        self,
        blade_parameters: Mapping[str, Any],
        *,
        near_wall_distance_m: float = 0.003,
    ) -> DeploymentPrediction:
        return self._predict_many(
            [blade_parameters],
            near_wall_distance_m=near_wall_distance_m,
            batch_size=1,
            metrics_only=False,
        )[0]

    def predict_many(
        self,
        blade_parameter_sets: Sequence[Mapping[str, Any]],
        *,
        near_wall_distance_m: float = 0.003,
        batch_size: int = 1,
    ) -> list[DeploymentPrediction]:
        return self._predict_many(
            blade_parameter_sets,
            near_wall_distance_m=near_wall_distance_m,
            batch_size=batch_size,
            metrics_only=False,
        )

    def predict_metrics_many(
        self,
        blade_parameter_sets: Sequence[Mapping[str, Any]],
        *,
        near_wall_distance_m: float = 0.003,
        batch_size: int = 1,
    ) -> list[HydraulicMetrics]:
        return self._predict_many(
            blade_parameter_sets,
            near_wall_distance_m=near_wall_distance_m,
            batch_size=batch_size,
            metrics_only=True,
        )


def _constraint_dominates(
    objectives_a: np.ndarray,
    violation_a: float,
    objectives_b: np.ndarray,
    violation_b: float,
) -> bool:
    feasible_a = math.isfinite(violation_a) and violation_a <= 1e-12
    feasible_b = math.isfinite(violation_b) and violation_b <= 1e-12
    if feasible_a != feasible_b:
        return feasible_a
    if not feasible_a:
        return violation_a < violation_b
    return bool(np.all(objectives_a <= objectives_b) and np.any(objectives_a < objectives_b))


def non_dominated_sort(objectives: np.ndarray, violations: np.ndarray) -> list[list[int]]:
    count = int(objectives.shape[0])
    dominates: list[list[int]] = [[] for _ in range(count)]
    dominated_count = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]
    for i in range(count):
        for j in range(i + 1, count):
            i_dominates = _constraint_dominates(objectives[i], float(violations[i]), objectives[j], float(violations[j]))
            j_dominates = _constraint_dominates(objectives[j], float(violations[j]), objectives[i], float(violations[i]))
            if i_dominates:
                dominates[i].append(j)
                dominated_count[j] += 1
            elif j_dominates:
                dominates[j].append(i)
                dominated_count[i] += 1
    fronts[0] = [index for index in range(count) if dominated_count[index] == 0]
    rank = 0
    while rank < len(fronts) and fronts[rank]:
        next_front: list[int] = []
        for i in fronts[rank]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        rank += 1
    return fronts


def _crowding_distance(front: Sequence[int], objectives: np.ndarray) -> dict[int, float]:
    if not front:
        return {}
    distance = {int(index): 0.0 for index in front}
    if len(front) <= 2:
        return {int(index): float("inf") for index in front}
    for objective_index in range(objectives.shape[1]):
        ordered = sorted(front, key=lambda index: objectives[index, objective_index])
        distance[int(ordered[0])] = distance[int(ordered[-1])] = float("inf")
        minimum = objectives[ordered[0], objective_index]
        maximum = objectives[ordered[-1], objective_index]
        span = maximum - minimum
        if not math.isfinite(float(span)) or span <= 1e-15:
            continue
        for position in range(1, len(ordered) - 1):
            index = int(ordered[position])
            distance[index] += float(
                (objectives[ordered[position + 1], objective_index] - objectives[ordered[position - 1], objective_index])
                / span
            )
    return distance


def _select_population(candidates: Sequence[OptimizationCandidate], size: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    objectives = np.asarray([candidate.objectives for candidate in candidates], dtype=float)
    violations = np.asarray([candidate.constraint_violation for candidate in candidates], dtype=float)
    fronts = non_dominated_sort(objectives, violations)
    selected: list[int] = []
    ranks = np.full(len(candidates), fill_value=len(candidates), dtype=int)
    crowding = np.zeros(len(candidates), dtype=float)
    for rank, front in enumerate(fronts):
        for index in front:
            ranks[index] = rank
        distances = _crowding_distance(front, objectives)
        for index, value in distances.items():
            crowding[index] = value
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
        else:
            selected.extend(sorted(front, key=lambda index: crowding[index], reverse=True)[:remaining])
            break
    return selected, ranks, crowding


def _latin_hypercube(rng: np.random.Generator, population_size: int, dimensions: int) -> np.ndarray:
    samples = np.empty((population_size, dimensions), dtype=float)
    for dimension in range(dimensions):
        samples[:, dimension] = (rng.permutation(population_size) + rng.random(population_size)) / population_size
    return samples


def optimize_blade_design(
    engine: DeploymentEngine,
    baseline_parameters: Mapping[str, Any],
    *,
    target_head_m: float = 0.1,
    variation_fraction: float = 0.05,
    near_wall_distance_m: float = 0.003,
    population_size: int = 8,
    generations: int = 3,
    feature_metric: str = "mixed",
    efficiency_weight: float = 0.5,
    inference_batch_size: int = 1,
    seed: int = 42,
    progress_callback: Callable[[int, int, str], None] | None = None,
    optimization_variables: Sequence[ShapeVariable] | None = None,
) -> OptimizationResult:
    """Run a compact constrained NSGA-II search over local blade shapes."""

    if target_head_m <= 0.0:
        raise ValueError("Target head must be positive.")
    if not 0.0 < variation_fraction <= 0.50:
        raise ValueError("variation_fraction must be within (0, 0.50].")
    population_size = max(4, int(population_size))
    generations = max(1, int(generations))
    efficiency_weight = float(np.clip(efficiency_weight, 0.0, 1.0))
    metric_attribute = {
        "mixed": "feature_velocity_m_s",
        "mean": "relative_velocity_mean_m_s",
        "max": "relative_velocity_max_m_s",
        "p95": "relative_velocity_p95_m_s",
        "rms": "relative_velocity_rms_m_s",
    }.get(feature_metric)
    if metric_attribute is None:
        raise ValueError("feature_metric must be mixed, mean, max, p95, or rms.")

    variables = (
        list(optimization_variables)
        if optimization_variables is not None
        else build_optimization_variables(baseline_parameters, variation_fraction)
    )
    if not variables:
        raise ValueError("Optimization requires at least one non-constant design variable.")
    lower = np.asarray([variable.lower for variable in variables], dtype=float)
    upper = np.asarray([variable.upper for variable in variables], dtype=float)
    baseline_vector = np.asarray([variable.baseline for variable in variables], dtype=float)
    rng = np.random.default_rng(seed)
    unit = _latin_hypercube(rng, population_size, len(variables))
    population_vectors = lower + unit * (upper - lower)
    population_vectors[0] = baseline_vector
    total_evaluations = population_size * generations
    completed = 0
    all_candidates: list[OptimizationCandidate] = []

    def evaluate(vectors: np.ndarray, generation: int) -> list[OptimizationCandidate]:
        nonlocal completed
        parameter_sets: list[Mapping[str, Any] | None] = []
        metric_results: list[HydraulicMetrics | Exception | None] = []
        valid_indices: list[int] = []
        for index, vector in enumerate(vectors):
            try:
                parameter_sets.append(apply_shape_vector(baseline_parameters, variables, vector))
                metric_results.append(None)
                valid_indices.append(index)
            except Exception as exc:
                parameter_sets.append(None)
                metric_results.append(exc)
        chunk_size = max(1, int(inference_batch_size))
        for start in range(0, len(valid_indices), chunk_size):
            chunk_indices = valid_indices[start : start + chunk_size]
            chunk = [parameter_sets[index] for index in chunk_indices]
            try:
                chunk_metrics = engine.predict_metrics_many(
                    chunk,
                    near_wall_distance_m=near_wall_distance_m,
                    batch_size=chunk_size,
                )
                for index, metrics in zip(chunk_indices, chunk_metrics):
                    metric_results[index] = metrics
            except Exception:
                for index, parameters in zip(chunk_indices, chunk):
                    try:
                        metric_results[index] = engine.predict_metrics_many(
                            [parameters],
                            near_wall_distance_m=near_wall_distance_m,
                            batch_size=1,
                        )[0]
                    except Exception as exc:  # candidate-level geometry/model failure
                        metric_results[index] = exc
            completed += len(chunk_indices)
            if progress_callback is not None:
                progress_callback(completed, total_evaluations, f"第 {generation + 1}/{generations} 代")

        invalid_count = len(vectors) - len(valid_indices)
        completed += invalid_count

        evaluated: list[OptimizationCandidate] = []
        for vector, parameters, result in zip(vectors, parameter_sets, metric_results):
            if isinstance(result, Exception):
                evaluated.append(
                    OptimizationCandidate(
                        values=tuple(float(value) for value in vector),
                        blade_parameters=parameters or copy.deepcopy(dict(baseline_parameters)),
                        metrics=None,
                        objectives=(float("inf"), float("inf")),
                        constraint_violation=float("inf"),
                        generation=generation,
                        error=str(result),
                    )
                )
                continue
            efficiency = float(result.hydraulic_efficiency)
            speed = float(getattr(result, metric_attribute))
            head = float(result.total_head_m)
            valid = all(math.isfinite(value) for value in (efficiency, speed, head))
            if valid:
                head_shortfall = max(float(target_head_m) - head, 0.0)
                violation = head_shortfall
            else:
                violation = float("inf")
            evaluated.append(
                OptimizationCandidate(
                    values=tuple(float(value) for value in vector),
                    blade_parameters=parameters,
                    metrics=result,
                    objectives=(-efficiency, speed) if valid else (float("inf"), float("inf")),
                    constraint_violation=float(violation),
                    generation=generation,
                    error=None if valid else "Non-finite performance metric",
                )
            )
        return evaluated

    population = evaluate(population_vectors, generation=0)
    all_candidates.extend(population)
    for generation in range(1, generations):
        selected, ranks, crowding = _select_population(population, population_size)
        parent_pool = [population[index] for index in selected]
        parent_ranks = np.asarray([ranks[index] for index in selected], dtype=int)
        parent_crowding = np.asarray([crowding[index] for index in selected], dtype=float)

        def tournament() -> np.ndarray:
            a, b = rng.integers(0, len(parent_pool), size=2)
            if parent_ranks[a] < parent_ranks[b] or (
                parent_ranks[a] == parent_ranks[b] and parent_crowding[a] >= parent_crowding[b]
            ):
                return np.asarray(parent_pool[a].values, dtype=float)
            return np.asarray(parent_pool[b].values, dtype=float)

        offspring_vectors = np.empty((population_size, len(variables)), dtype=float)
        for child_index in range(population_size):
            parent_a, parent_b = tournament(), tournament()
            alpha = rng.uniform(-0.1, 1.1, size=len(variables))
            child = alpha * parent_a + (1.0 - alpha) * parent_b
            mutation_mask = rng.random(len(variables)) < (1.0 / len(variables))
            child[mutation_mask] += rng.normal(0.0, 0.10, size=int(np.sum(mutation_mask))) * (upper - lower)[mutation_mask]
            offspring_vectors[child_index] = np.clip(child, lower, upper)
        offspring = evaluate(offspring_vectors, generation=generation)
        all_candidates.extend(offspring)
        combined = population + offspring
        survivor_indices, _, _ = _select_population(combined, population_size)
        population = [combined[index] for index in survivor_indices]

    objectives = np.asarray([candidate.objectives for candidate in all_candidates], dtype=float)
    violations = np.asarray([candidate.constraint_violation for candidate in all_candidates], dtype=float)
    fronts = non_dominated_sort(objectives, violations)
    feasible_indices = [index for index, candidate in enumerate(all_candidates) if candidate.feasible]
    if feasible_indices:
        first_front = fronts[0] if fronts else []
        pareto_indices = [index for index in first_front if all_candidates[index].feasible]
        if not pareto_indices:
            feasible_objectives = objectives[feasible_indices]
            feasible_violations = np.zeros(len(feasible_indices), dtype=float)
            local_front = non_dominated_sort(feasible_objectives, feasible_violations)[0]
            pareto_indices = [feasible_indices[index] for index in local_front]
        efficiencies = np.asarray([all_candidates[index].metrics.hydraulic_efficiency for index in pareto_indices], dtype=float)
        speeds = np.asarray([getattr(all_candidates[index].metrics, metric_attribute) for index in pareto_indices], dtype=float)
        eta_loss = (np.max(efficiencies) - efficiencies) / max(float(np.ptp(efficiencies)), 1e-12)
        speed_loss = (speeds - np.min(speeds)) / max(float(np.ptp(speeds)), 1e-12)
        half_span = np.maximum(0.5 * (upper - lower), 1e-12)
        deviations = np.asarray(
            [np.sqrt(np.mean(((np.asarray(all_candidates[index].values) - baseline_vector) / half_span) ** 2)) for index in pareto_indices]
        )
        scores = efficiency_weight * eta_loss + (1.0 - efficiency_weight) * speed_loss + 0.05 * deviations
        recommended_index = pareto_indices[int(np.argmin(scores))]
        message = f"找到 {len(feasible_indices)} 个满足扬程约束的设计，返回 {len(pareto_indices)} 个 Pareto 解。"
    else:
        finite_indices = [
            index
            for index, candidate in enumerate(all_candidates)
            if candidate.metrics is not None and math.isfinite(candidate.constraint_violation)
        ]
        pareto_indices = []
        recommended_index = min(finite_indices, key=lambda index: all_candidates[index].constraint_violation) if finite_indices else None
        best_head = (
            all_candidates[recommended_index].metrics.total_head_m
            if recommended_index is not None and all_candidates[recommended_index].metrics is not None
            else float("nan")
        )
        message = (
            f"当前局部设计域内没有方案达到 {target_head_m:.3f} m；"
            f"扫描到的最高扬程为 {best_head:.3f} m。请扩大训练数据覆盖后再优化，不会伪造可行解。"
        )

    return OptimizationResult(
        variables=variables,
        candidates=all_candidates,
        pareto_indices=pareto_indices,
        recommended_index=recommended_index,
        target_head_m=float(target_head_m),
        feature_metric=feature_metric,
        total_evaluations=len(all_candidates),
        feasible_count=len(feasible_indices),
        message=message,
    )


def prediction_to_npz_bytes(prediction: DeploymentPrediction) -> bytes:
    theta = prediction.coordinates.theta_rad[None, :, None]
    cosine = np.cos(theta)
    sine = np.sin(theta)
    ux = prediction.fields_physical["UR"] * cosine - prediction.fields_physical["UT"] * sine
    uy = prediction.fields_physical["UR"] * sine + prediction.fields_physical["UT"] * cosine
    metadata = {
        "case": {key: value for key, value in prediction.case.items() if key != "blade_params"},
        "blade_parameters": prediction.blade_parameters,
        "metrics": prediction.metrics.to_record(),
        "checkpoint_path": prediction.checkpoint_path,
        "geometry_hashes": {
            "blade_mask_sha256": hashlib.sha256(np.ascontiguousarray(prediction.blade_mask).tobytes()).hexdigest(),
            "phi_sha256": hashlib.sha256(np.ascontiguousarray(prediction.phi).tobytes()).hexdigest(),
        },
        "coordinate_mapping": {
            "r": "rh + R*(rs-rh)",
            "theta": "Theta*2*pi/n_blade",
            "z": "z0 + Z*h",
            "x": "r*cos(theta)",
            "y": "r*sin(theta)",
        },
        "compatibility_migration": prediction.compatibility_migration,
    }
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        r_m=prediction.coordinates.r_m,
        theta_rad=prediction.coordinates.theta_rad,
        z_m=prediction.coordinates.z_m,
        x_m=prediction.coordinates.x_m,
        y_m=prediction.coordinates.y_m,
        UR_m_s=prediction.fields_physical["UR"],
        UT_m_s=prediction.fields_physical["UT"],
        UZ_m_s=prediction.fields_physical["UZ"],
        Ux_m_s=ux,
        Uy_m_s=uy,
        P_pa=prediction.fields_physical["P"],
        UR_dimensionless=prediction.fields_dimensionless["UR"],
        UT_dimensionless=prediction.fields_dimensionless["UT"],
        UZ_dimensionless=prediction.fields_dimensionless["UZ"],
        P_dimensionless=prediction.fields_dimensionless["P"],
        relative_speed_m_s=prediction.relative_speed_m_s,
        wall_distance_m=prediction.wall_distance_m,
        blade_mask=prediction.blade_mask,
        phi=prediction.phi,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
    )
    return buffer.getvalue()


def candidate_rows(result: OptimizationResult) -> list[dict[str, Any]]:
    pareto_set = set(result.pareto_indices)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(result.candidates):
        metrics = candidate.metrics
        rows.append(
            {
                "index": index,
                "generation": candidate.generation + 1,
                "feasible": candidate.feasible,
                "pareto": index in pareto_set,
                "head_m": None if metrics is None else metrics.total_head_m,
                "efficiency": None if metrics is None else metrics.hydraulic_efficiency,
                "feature_velocity_m_s": None if metrics is None else metrics.feature_velocity_m_s,
                "mean_velocity_m_s": None if metrics is None else metrics.relative_velocity_mean_m_s,
                "max_velocity_m_s": None if metrics is None else metrics.relative_velocity_max_m_s,
                "torque_n_m": None if metrics is None else metrics.driving_torque_n_m,
                "pressure_torque_n_m": None if metrics is None else metrics.blade_pressure_torque_n_m,
                "angular_momentum_torque_n_m": None if metrics is None else metrics.angular_momentum_torque_n_m,
                "geometry_trust": None if metrics is None else metrics.geometry_trust_label,
                "mass_imbalance": None if metrics is None else metrics.mass_imbalance_ratio,
                "head_shortfall_m": (
                    None if metrics is None else max(result.target_head_m - metrics.total_head_m, 0.0)
                ),
                "constraint_violation": candidate.constraint_violation,
                "error": candidate.error,
            }
        )
    return rows
