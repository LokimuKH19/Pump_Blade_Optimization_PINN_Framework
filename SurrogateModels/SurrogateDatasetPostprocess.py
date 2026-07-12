# -*- coding: utf-8 -*-
"""Folder-oriented CFD/surrogate flow-field post-processing.

The public entry points are :func:`postprocess_dataset_folder` and
:func:`postprocess_dataset_folders`.  They are intentionally UI agnostic so a
Streamlit page can cache the returned dataclasses and render them directly.

Supported field sources, in priority order:

1. ``.npz`` exported by ``SurrogateDeployment.prediction_to_npz_bytes``
   (``UR_m_s``/``UT_m_s``/``UZ_m_s``/``P_pa``);
2. generator ``.npz`` physical fields
   (``u_r``/``u_theta_absolute``/``u_z``/``p``);
3. generator ``.npz`` dimensionless fields (``UR``/``UT``/``UZ``/``P``),
   converted with discovered or deployment-consistent scales;
4. a Fluent-style CSV next to ``blade_params.json``, interpolated through the
   existing ``SurrogateModelingData`` loader.

Hydraulic quantities are computed by ``SurrogateDeployment`` itself.  This is
important: the dataset view and the deployment view therefore share exactly
the same definitions for head, pressure rise, torque, power, efficiency and
near-wall feature velocity.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from BladeImport import PassageGeometry, build_blade_boundary
from SurrogateDeployment import HydraulicMetrics, compute_hydraulic_metrics


ROOT = Path(__file__).resolve().parent
FIELD_NAMES = ("UR", "UT", "UZ", "P")
DEFAULT_GENERATOR_CASE = {
    "rh": 0.0605,
    "rs": 0.08,
    "h": 0.125,
    "z0": 0.0,
    "mu": 0.006,
    "rho": 10650.0,
    "omega": -420.0 * math.pi / 60.0,
    "qv": 0.16,
    "n_blade": 6,
    "g": 9.8,
    "absolute_frame": True,
}
DEFAULT_CSV_CASE = {**DEFAULT_GENERATOR_CASE, "omega": -210.0 * 2.0 * math.pi / 60.0, "qv": 0.025}


@dataclass(frozen=True)
class PostprocessDiagnostic:
    """One machine-readable diagnostic emitted during discovery or analysis."""

    severity: str
    code: str
    message: str
    path: str | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetPostprocessResult:
    """Structured result for one input record.

    ``fields_physical`` always uses SI units when present: m/s for velocity and
    Pa for pressure.  A failed input is returned as a record with ``metrics``
    set to ``None`` and one or more ``error`` diagnostics; batch callers do not
    lose the other requested folders.
    """

    record_index: int
    requested_path: str
    resolved_path: Path
    duplicate: bool = False
    duplicate_of_index: int | None = None
    source_kind: str | None = None
    field_path: Path | None = None
    condition_path: Path | None = None
    blade_path: Path | None = None
    case: dict[str, Any] = field(default_factory=dict)
    fields_physical: dict[str, np.ndarray] | None = None
    blade_mask: np.ndarray | None = None
    phi: np.ndarray | None = None
    metrics: HydraulicMetrics | None = None
    diagnostics: list[PostprocessDiagnostic] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.metrics is not None and not any(item.severity == "error" for item in self.diagnostics)

    def to_summary_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "record_index": self.record_index,
            "requested_path": self.requested_path,
            "resolved_path": str(self.resolved_path),
            "duplicate": self.duplicate,
            "duplicate_of_index": self.duplicate_of_index,
            "success": self.success,
            "source_kind": self.source_kind,
            "field_path": None if self.field_path is None else str(self.field_path),
            "condition_path": None if self.condition_path is None else str(self.condition_path),
            "blade_path": None if self.blade_path is None else str(self.blade_path),
            "diagnostic_count": len(self.diagnostics),
            "errors": "；".join(item.message for item in self.diagnostics if item.severity == "error"),
            "warnings": "；".join(item.message for item in self.diagnostics if item.severity == "warning"),
        }
        if self.metrics is not None:
            record.update(self.metrics.to_record())
        return record


@dataclass
class DatasetBatchPostprocessResult:
    results: list[DatasetPostprocessResult]
    near_wall_distance_m: float

    @property
    def success_count(self) -> int:
        return sum(item.success for item in self.results)

    @property
    def duplicate_count(self) -> int:
        return sum(item.duplicate for item in self.results)

    def summary_records(self) -> list[dict[str, Any]]:
        return [item.to_summary_record() for item in self.results]


def _diag(
    diagnostics: list[PostprocessDiagnostic],
    severity: str,
    code: str,
    message: str,
    path: str | Path | None = None,
) -> None:
    diagnostics.append(
        PostprocessDiagnostic(
            severity=str(severity),
            code=str(code),
            message=str(message),
            path=None if path is None else str(path),
        )
    )


def _read_json_documents(folder: Path, diagnostics: list[PostprocessDiagnostic]) -> dict[Path, dict[str, Any]]:
    documents: dict[Path, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json"), key=lambda item: item.name.lower()):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _diag(diagnostics, "warning", "JSON_UNREADABLE", f"忽略无法读取的 JSON：{exc}", path)
            continue
        if isinstance(value, dict):
            documents[path.resolve()] = value
    return documents


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return
    yield value
    for child in value.values():
        if isinstance(child, Mapping):
            yield from _iter_mappings(child)


def _first_value(sources: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Any:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _as_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_reference(raw: str | Path, folder: Path) -> Path | None:
    raw_path = Path(str(raw))
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend([folder / raw_path, ROOT / raw_path, ROOT.parent / raw_path])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def _discover_blade_parameters(
    folder: Path,
    documents: Mapping[Path, Mapping[str, Any]],
    embedded_metadata: Mapping[str, Any],
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[dict[str, Any] | None, Path | None]:
    embedded = embedded_metadata.get("blade_parameters")
    if isinstance(embedded, Mapping):
        return copy.deepcopy(dict(embedded)), None

    direct = folder / "blade_params.json"
    if direct.exists():
        try:
            value = json.loads(direct.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value, direct.resolve()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _diag(diagnostics, "warning", "BLADE_JSON_UNREADABLE", f"叶型 JSON 无法读取：{exc}", direct)

    references: list[Any] = []
    case_metadata = embedded_metadata.get("case")
    if isinstance(case_metadata, Mapping):
        references.append(case_metadata.get("blade_params"))
    for document in documents.values():
        files = document.get("files")
        if isinstance(files, Mapping):
            references.append(files.get("blade_params"))
        shape = document.get("shape_parameters")
        if isinstance(shape, Mapping):
            references.append(shape.get("blade_params"))
        references.append(document.get("blade_params"))

    for raw in references:
        if raw is None or isinstance(raw, Mapping):
            continue
        resolved = _resolve_reference(raw, folder)
        if resolved is None:
            continue
        try:
            value = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("blade_layers"), list):
            return value, resolved

    _diag(
        diagnostics,
        "warning",
        "BLADE_JSON_MISSING",
        "未发现可用叶型 JSON；若场文件包含 phi，将由 phi 近似恢复叶片掩膜。",
        folder,
    )
    return None, None


def _npz_source_score(path: Path) -> tuple[int, str]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            keys = set(payload.files)
    except Exception:
        return -1, ""
    if {"UR_m_s", "UT_m_s", "UZ_m_s", "P_pa"} <= keys:
        score, kind = 300, "surrogate_npz_physical"
    elif {"u_z", "p"} <= keys and ({"u_theta", "u_theta_absolute"} & keys):
        score, kind = 240, "generator_npz_physical"
    elif {"UZ", "P"} <= keys and ({"UT", "UTheta", "UTheta_absolute"} & keys):
        score, kind = 180, "generator_npz_dimensionless"
    else:
        return -1, ""
    lower_name = path.name.lower()
    if "flow_field" in lower_name:
        score += 20
    if "physical" in lower_name or "prediction" in lower_name:
        score += 10
    return score, kind


def _discover_npz(folder: Path, diagnostics: list[PostprocessDiagnostic]) -> tuple[Path | None, str | None]:
    candidates: list[tuple[int, float, str, Path, str]] = []
    for path in folder.glob("*.npz"):
        score, kind = _npz_source_score(path)
        if score < 0:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        candidates.append((score, modified, path.name.lower(), path.resolve(), kind))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    chosen = candidates[0]
    if len(candidates) > 1:
        _diag(
            diagnostics,
            "warning",
            "MULTIPLE_FIELD_FILES",
            f"发现 {len(candidates)} 个可识别 NPZ，已按物理字段优先级选择 {chosen[3].name}。",
            folder,
        )
    return chosen[3], chosen[4]


def _csv_is_flow_field(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            row = next(csv.reader(handle), [])
    except OSError:
        return False
    names = {str(value).strip().lower().replace("_", "-") for value in row}
    has_pressure = any("pressure" in value or value == "p" for value in names)
    has_coordinate = any("coordinate" in value for value in names)
    has_velocity = sum("velocity" in value for value in names) >= 2
    return has_pressure and has_coordinate and has_velocity


def _discover_csv(folder: Path, diagnostics: list[PostprocessDiagnostic]) -> Path | None:
    candidates = [path.resolve() for path in folder.glob("*.csv") if _csv_is_flow_field(path)]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (-path.stat().st_size, path.name.lower()))
    if len(candidates) > 1:
        _diag(
            diagnostics,
            "warning",
            "MULTIPLE_CFD_CSV",
            f"发现 {len(candidates)} 个 CFD CSV，已选择数据量最大的 {candidates[0].name}。",
            folder,
        )
    return candidates[0]


def _load_npz_selected(path: Path, diagnostics: list[PostprocessDiagnostic]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    wanted = {
        "UR_m_s", "UT_m_s", "UZ_m_s", "P_pa",
        "u_r", "u_theta", "u_theta_absolute", "u_z", "p",
        "UR", "UT", "UTheta", "UTheta_absolute", "UZ", "P",
        "blade_mask", "phi", "absolute_frame", "metadata_json",
        "u_omega", "u_zo", "P0", "history_momentum", "history_continuity",
        "history_update", "history_q_hat",
    }
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in payload.files:
            if key not in wanted:
                continue
            value = np.asarray(payload[key])
            if key == "metadata_json":
                try:
                    parsed = json.loads(str(value.item()))
                    if isinstance(parsed, dict):
                        metadata.update(parsed)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    _diag(diagnostics, "warning", "NPZ_METADATA_INVALID", f"NPZ metadata_json 无法解析：{exc}", path)
            elif value.ndim == 0:
                metadata[key] = _as_scalar(value)
            elif key.startswith("history_"):
                if value.size:
                    metadata[f"{key}_final"] = _as_scalar(value.reshape(-1)[-1])
            else:
                arrays[key] = value.copy()
    return arrays, metadata


def _ordered_sources(
    documents: Mapping[Path, Mapping[str, Any]],
    embedded_metadata: Mapping[str, Any],
    blade_parameters: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = []
    case = embedded_metadata.get("case")
    if isinstance(case, Mapping):
        sources.append(case)

    ordered_documents = sorted(
        documents.items(),
        key=lambda item: (
            0 if item[0].name == "condition_parameters.json" else
            1 if item[0].name == "shape_parameters.json" else
            2,
            item[0].name.lower(),
        ),
    )
    for _, document in ordered_documents:
        for key in ("condition_parameters", "physics_config", "shape_parameters", "dimensionless_parameters"):
            child = document.get(key)
            if isinstance(child, Mapping):
                sources.append(child)
        sources.extend(_iter_mappings(document))

    if blade_parameters is not None:
        global_parameters = blade_parameters.get("global_parameters")
        if isinstance(global_parameters, Mapping):
            sources.append(global_parameters)
    sources.append(embedded_metadata)
    return sources


def _build_case_and_scales(
    *,
    shape: tuple[int, int, int],
    source_kind: str,
    documents: Mapping[Path, Mapping[str, Any]],
    embedded_metadata: Mapping[str, Any],
    blade_parameters: Mapping[str, Any] | None,
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[dict[str, Any], dict[str, float]]:
    if len(set(shape)) != 1:
        raise ValueError(f"现有部署指标要求立方网格，实际字段 shape={shape}。")
    sources = _ordered_sources(documents, embedded_metadata, blade_parameters)
    aliases = {
        "rh": ("rh", "hub_radius"),
        "rs": ("rs", "shroud_radius"),
        "h": ("h", "passage_height", "passage_height_H1"),
        "z0": ("z0", "passage_z0"),
        "mu": ("mu", "dynamic_viscosity"),
        "rho": ("rho", "rho_physical", "density"),
        "omega": ("omega",),
        "qv": ("qv", "qv_reference_total_m3s", "prescribed_flow_rate_m3_s"),
        "n_blade": ("n_blade", "n_blades", "blade_count", "blade_count_N"),
        "g": ("g", "gravity"),
        "absolute_frame": ("absolute_frame",),
    }
    defaults = DEFAULT_CSV_CASE if source_kind == "cfd_csv" else DEFAULT_GENERATOR_CASE
    case: dict[str, Any] = {"n": int(shape[0])}
    defaulted: list[str] = []
    for key, keys in aliases.items():
        value = _first_value(sources, keys)
        if value is None and key == "omega":
            rpm = _first_value(sources, ("rpm", "RPM"))
            if rpm is not None:
                value = float(rpm) * 2.0 * math.pi / 60.0
        if value is None:
            value = defaults[key]
            defaulted.append(key)
        if key in {"n_blade"}:
            case[key] = int(value)
        elif key == "absolute_frame":
            case[key] = bool(value)
        else:
            case[key] = float(value)
    if defaulted:
        _diag(
            diagnostics,
            "warning",
            "CASE_DEFAULTS_USED",
            "工况文件缺少以下字段，采用本仓库生成器默认值：" + ", ".join(defaulted) + "。",
        )

    if not (case["rs"] > case["rh"] > 0.0 and case["h"] > 0.0 and case["rho"] > 0.0 and case["n_blade"] > 0):
        raise ValueError("工况几何或物性无效：要求 rs>rh>0、h>0、rho>0、n_blade>0。")

    u_omega_exact = _first_value(sources, ("u_omega",))
    u_zo_exact = _first_value(sources, ("u_zo", "u_zo_reference"))
    p0_exact = _first_value(sources, ("P0", "P0_pa", "pressure_scale_pa"))
    u_omega = float(u_omega_exact) if u_omega_exact is not None else max(abs(case["rs"] * case["omega"]), 1e-12)
    u_zo = (
        float(u_zo_exact)
        if u_zo_exact is not None
        else case["qv"] / (math.pi * (case["rs"] ** 2 - case["rh"] ** 2))
    )
    p0 = (
        float(p0_exact)
        if p0_exact is not None
        else case["rho"] * (u_zo**2 + 0.5 * u_omega**2 + case["g"] * case["h"])
    )
    return case, {"u_omega": u_omega, "u_zo": u_zo, "P0": p0}


def _as_3d(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 3:
        raise ValueError(f"字段 {name} 必须是三维数组，实际 shape={array.shape}。")
    return array.astype(np.float32, copy=False)


def _field_shape_from_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    preferred = (
        "UZ_m_s", "u_z", "UZ", "P_pa", "p", "P", "phi", "blade_mask",
    )
    for key in preferred:
        if key in arrays:
            return tuple(int(value) for value in _as_3d(arrays[key], key).shape)
    raise ValueError("场文件没有可识别的三维速度、压力、phi 或 blade_mask。")


def _extract_physical_fields(
    arrays: Mapping[str, np.ndarray],
    case: dict[str, Any],
    scales: Mapping[str, float],
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[dict[str, np.ndarray], str]:
    shape = _field_shape_from_arrays(arrays)

    def zeros() -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    if all(key in arrays for key in ("UR_m_s", "UT_m_s", "UZ_m_s", "P_pa")):
        fields = {
            "UR": _as_3d(arrays["UR_m_s"], "UR_m_s"),
            "UT": _as_3d(arrays["UT_m_s"], "UT_m_s"),
            "UZ": _as_3d(arrays["UZ_m_s"], "UZ_m_s"),
            "P": _as_3d(arrays["P_pa"], "P_pa"),
        }
        field_basis = "explicit_si"
        frame_is_absolute = bool(case.get("absolute_frame", True))
    elif "u_z" in arrays and "p" in arrays and ("u_theta_absolute" in arrays or "u_theta" in arrays):
        ur = _as_3d(arrays["u_r"], "u_r") if "u_r" in arrays else zeros()
        if "u_r" not in arrays:
            _diag(diagnostics, "info", "UR_ASSUMED_ZERO", "场文件未保存径向速度，按零场处理。")
        theta_key = "u_theta_absolute" if "u_theta_absolute" in arrays else "u_theta"
        fields = {
            "UR": ur,
            "UT": _as_3d(arrays[theta_key], theta_key),
            "UZ": _as_3d(arrays["u_z"], "u_z"),
            "P": _as_3d(arrays["p"], "p"),
        }
        field_basis = "generator_si"
        frame_is_absolute = theta_key == "u_theta_absolute" or bool(case.get("absolute_frame", True))
    elif "UZ" in arrays and "P" in arrays and any(key in arrays for key in ("UTheta_absolute", "UT", "UTheta")):
        theta_key = next(key for key in ("UTheta_absolute", "UT", "UTheta") if key in arrays)
        fields = {
            "UR": (_as_3d(arrays["UR"], "UR") if "UR" in arrays else zeros()) * float(scales["u_omega"]),
            "UT": _as_3d(arrays[theta_key], theta_key) * float(scales["u_omega"]),
            "UZ": _as_3d(arrays["UZ"], "UZ") * float(scales["u_zo"]),
            "P": _as_3d(arrays["P"], "P") * float(scales["P0"]),
        }
        if "UR" not in arrays:
            _diag(diagnostics, "info", "UR_ASSUMED_ZERO", "无量纲场未保存 UR，按零场处理。")
        field_basis = "dimensionless_scaled"
        frame_is_absolute = theta_key == "UTheta_absolute" or bool(case.get("absolute_frame", True))
        _diag(
            diagnostics,
            "info",
            "DIMENSIONLESS_TO_SI",
            "已按 u_omega、u_zo 和 P0 将无量纲场映射到 SI。",
        )
    else:
        raise ValueError("无法从 NPZ 组合出 UR、UT、UZ、P 四个物理场。")

    expected = fields["UZ"].shape
    if any(array.shape != expected for array in fields.values()):
        raise ValueError("UR、UT、UZ、P 的网格 shape 不一致。")

    if not frame_is_absolute:
        radii = np.linspace(case["rh"], case["rs"], expected[0], dtype=np.float32)
        wall_velocity = (float(case["omega"]) * radii)[:, None, None]
        fields["UT"] = fields["UT"] + wall_velocity
        _diag(
            diagnostics,
            "info",
            "ROTATING_TO_ABSOLUTE",
            "周向速度已由旋转参考系转换到绝对参考系。",
        )
    case["absolute_frame"] = True

    for name, array in fields.items():
        finite = np.isfinite(array)
        if not np.all(finite):
            fraction = 1.0 - float(np.count_nonzero(finite)) / max(array.size, 1)
            _diag(diagnostics, "warning", "NONFINITE_FIELD", f"{name} 含 {fraction:.3%} 非有限值。")
    return fields, field_basis


def _build_mask_and_phi(
    arrays: Mapping[str, np.ndarray],
    case: Mapping[str, Any],
    blade_parameters: Mapping[str, Any] | None,
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        shape = _field_shape_from_arrays(arrays)
    except ValueError:
        if blade_parameters is None or case.get("n") is None:
            raise
        shape = (int(case["n"]),) * 3
    phi = _as_3d(arrays["phi"], "phi") if "phi" in arrays else None
    explicit_mask = _as_3d(arrays["blade_mask"], "blade_mask") > 0.5 if "blade_mask" in arrays else None

    rebuilt_mask: np.ndarray | None = None
    if blade_parameters is not None:
        try:
            boundary = build_blade_boundary(
                blade_parameters,
                PassageGeometry(
                    hub_radius=float(case["rh"]),
                    shroud_radius=float(case["rs"]),
                    passage_height=float(case["h"]),
                    blade_count=int(case["n_blade"]),
                    grid_size=int(shape[0]),
                    passage_z0=float(case.get("z0", 0.0)),
                ),
            )
            rebuilt_mask = np.asarray(boundary.mask, dtype=bool)
        except Exception as exc:
            _diag(diagnostics, "warning", "BLADE_REBUILD_FAILED", f"叶型重建失败，退回场内掩膜：{exc}")

    if explicit_mask is not None:
        mask, source = explicit_mask, "blade_mask"
    elif rebuilt_mask is not None:
        mask, source = rebuilt_mask, "blade_json"
    elif phi is not None:
        mask, source = phi <= 0.05, "phi<=0.05"
        _diag(diagnostics, "warning", "MASK_FROM_PHI", "叶片掩膜由 phi<=0.05 近似恢复。")
    else:
        raise ValueError("既没有 blade_mask/phi，也无法从叶型 JSON 重建叶片。")

    if mask.shape != shape:
        raise ValueError(f"叶片掩膜 shape={mask.shape} 与流场 shape={shape} 不一致。")
    if not np.any(mask):
        raise ValueError("叶片掩膜为空，无法计算近壁特征流速。")
    if phi is None:
        phi = (~mask).astype(np.float32)
    elif phi.shape != shape:
        raise ValueError("phi 与流场 shape 不一致。")

    if rebuilt_mask is not None and source != "blade_json":
        intersection = np.count_nonzero(mask & rebuilt_mask)
        union = np.count_nonzero(mask | rebuilt_mask)
        iou = intersection / max(union, 1)
        if iou < 0.90:
            _diag(diagnostics, "warning", "MASK_BLADE_IOU_LOW", f"场内掩膜与叶型重建掩膜 IoU={iou:.3f}。")
    return mask, phi.astype(np.float32, copy=False), source


def _signed_solid_ut(case: Mapping[str, Any], shape: tuple[int, int, int]) -> np.ndarray:
    radii = np.linspace(float(case["rh"]), float(case["rs"]), shape[0], dtype=np.float32)
    sign = 1.0 if float(case["omega"]) >= 0.0 else -1.0
    profile = sign * radii / max(float(case["rs"]), 1e-12)
    return np.broadcast_to(profile[:, None, None], shape).astype(np.float32, copy=True)


def _convergence_diagnostics(
    documents: Mapping[Path, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    diagnostics: list[PostprocessDiagnostic],
) -> dict[str, Any]:
    all_mappings: list[Mapping[str, Any]] = []
    for document in documents.values():
        all_mappings.extend(_iter_mappings(document))
    all_mappings.append(metadata)
    finite = _first_value(all_mappings, ("finite",))
    converged = _first_value(all_mappings, ("converged", "log_converged"))
    final_mass = _first_value(all_mappings, ("final_mass", "history_continuity_final"))
    final_momentum = _first_value(all_mappings, ("final_momentum", "history_momentum_final"))
    if finite is False:
        _diag(diagnostics, "error", "SOURCE_NONFINITE", "源数据摘要标记 finite=false。")
    if converged is False:
        _diag(diagnostics, "warning", "SOURCE_NOT_CONVERGED", "源数据摘要标记为未收敛。")
    return {
        "finite": finite,
        "converged": converged,
        "final_mass": final_mass,
        "final_momentum": final_momentum,
    }


def _find_condition_path(documents: Mapping[Path, Mapping[str, Any]]) -> Path | None:
    for path in documents:
        if path.name == "condition_parameters.json":
            return path
    for path, document in documents.items():
        if isinstance(document.get("condition_parameters"), Mapping):
            return path
    return None


def _postprocess_npz(
    folder: Path,
    field_path: Path,
    source_kind: str,
    documents: Mapping[Path, Mapping[str, Any]],
    near_wall_distance_m: float,
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[
    dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray, HydraulicMetrics,
    Path | None, dict[str, Any], str,
]:
    arrays, embedded_metadata = _load_npz_selected(field_path, diagnostics)
    blade_parameters, blade_path = _discover_blade_parameters(folder, documents, embedded_metadata, diagnostics)
    shape = _field_shape_from_arrays(arrays)
    case, scales = _build_case_and_scales(
        shape=shape,
        source_kind=source_kind,
        documents=documents,
        embedded_metadata=embedded_metadata,
        blade_parameters=blade_parameters,
        diagnostics=diagnostics,
    )
    fields, field_basis = _extract_physical_fields(arrays, case, scales, diagnostics)
    mask, phi, mask_source = _build_mask_and_phi(arrays, case, blade_parameters, diagnostics)
    solid_ut = _signed_solid_ut(case, shape)
    metrics, _, _, _ = compute_hydraulic_metrics(
        fields,
        case,
        mask,
        solid_ut,
        u_omega=float(scales["u_omega"]),
        near_wall_distance_m=float(near_wall_distance_m),
    )
    metrics = replace(
        metrics,
        wall_speed_convention="signed_omega_r_dataset",
        notes=tuple(metrics.notes) + ("数据集后处理使用有符号 omega*r 叶面速度。",),
    )
    metadata = {
        "field_basis": field_basis,
        "mask_source": mask_source,
        "scales": dict(scales),
        "embedded_metadata": embedded_metadata,
        "convergence": _convergence_diagnostics(documents, embedded_metadata, diagnostics),
    }
    return case, fields, mask, phi, metrics, blade_path, metadata, field_basis


def _postprocess_csv(
    folder: Path,
    csv_path: Path,
    documents: Mapping[Path, Mapping[str, Any]],
    near_wall_distance_m: float,
    grid_size: int,
    interpolation_chunk_size: int,
    diagnostics: list[PostprocessDiagnostic],
) -> tuple[
    dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray, HydraulicMetrics,
    Path | None, dict[str, Any], str,
]:
    blade_parameters, blade_path = _discover_blade_parameters(folder, documents, {}, diagnostics)
    if blade_path is None or not (folder / "blade_params.json").exists():
        raise FileNotFoundError("Fluent CSV 后处理要求目录内存在 blade_params.json。")

    provisional_case, _ = _build_case_and_scales(
        shape=(int(grid_size),) * 3,
        source_kind="cfd_csv",
        documents=documents,
        embedded_metadata={},
        blade_parameters=blade_parameters,
        diagnostics=diagnostics,
    )
    from SurrogateModelingData import make_supervised_simulation_case

    case = make_supervised_simulation_case(
        folder,
        n=int(grid_size),
        mu=float(provisional_case["mu"]),
        rho=float(provisional_case["rho"]),
        omega=float(provisional_case["omega"]),
        qv=float(provisional_case["qv"]),
        g=float(provisional_case["g"]),
        csv_path=csv_path,
        interpolation_chunk_size=int(interpolation_chunk_size),
        verbose_geometry_check=False,
    )
    fields = {name: np.asarray(case[name], dtype=np.float32) for name in FIELD_NAMES}
    shape = tuple(int(value) for value in fields["UZ"].shape)
    arrays = {"phi": np.asarray(case.get("phi"))} if case.get("phi") is not None else {}
    mask, phi, mask_source = _build_mask_and_phi(arrays, case, blade_parameters, diagnostics)
    u_omega = max(abs(float(case["rs"]) * float(case["omega"])), 1e-12)
    solid_ut = _signed_solid_ut(case, shape)
    metrics, _, _, _ = compute_hydraulic_metrics(
        fields,
        case,
        mask,
        solid_ut,
        u_omega=u_omega,
        near_wall_distance_m=float(near_wall_distance_m),
    )
    metrics = replace(
        metrics,
        wall_speed_convention="signed_omega_r_dataset",
        notes=tuple(metrics.notes) + ("数据集后处理使用有符号 omega*r 叶面速度。",),
    )
    metadata = {
        "field_basis": "fluent_csv_si",
        "mask_source": mask_source,
        "csv_points_in_sector": case.get("csv_points_in_sector"),
        "convergence": _convergence_diagnostics(documents, {}, diagnostics),
    }
    return case, fields, mask, phi, metrics, blade_path, metadata, "fluent_csv_si"


def postprocess_dataset_folder(
    folder: str | Path,
    *,
    near_wall_distance_m: float = 0.003,
    grid_size: int = 64,
    interpolation_chunk_size: int = 250_000,
    record_index: int = 0,
) -> DatasetPostprocessResult:
    """Discover and post-process one CFD/surrogate dataset directory.

    Errors are represented in the returned result instead of being raised,
    except for programmer-level argument errors such as a non-positive ``d``.
    """

    if float(near_wall_distance_m) <= 0.0:
        raise ValueError("near_wall_distance_m must be positive.")
    if int(grid_size) < 8:
        raise ValueError("grid_size must be at least 8.")

    requested = str(folder)
    path = Path(folder).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    diagnostics: list[PostprocessDiagnostic] = []
    result = DatasetPostprocessResult(
        record_index=int(record_index),
        requested_path=requested,
        resolved_path=resolved,
        diagnostics=diagnostics,
    )
    if not resolved.exists() or not resolved.is_dir():
        _diag(diagnostics, "error", "DIRECTORY_NOT_FOUND", "输入路径不是可读取目录。", resolved)
        return result

    try:
        documents = _read_json_documents(resolved, diagnostics)
        result.condition_path = _find_condition_path(documents)
        npz_path, npz_kind = _discover_npz(resolved, diagnostics)
        if npz_path is not None and npz_kind is not None:
            result.source_kind = npz_kind
            result.field_path = npz_path
            (
                result.case,
                result.fields_physical,
                result.blade_mask,
                result.phi,
                result.metrics,
                result.blade_path,
                result.source_metadata,
                _,
            ) = _postprocess_npz(
                resolved,
                npz_path,
                npz_kind,
                documents,
                near_wall_distance_m,
                diagnostics,
            )
        else:
            csv_path = _discover_csv(resolved, diagnostics)
            if csv_path is None:
                raise FileNotFoundError("目录内没有可识别的 CFD/代理场 NPZ 或 Fluent CSV。")
            result.source_kind = "cfd_csv"
            result.field_path = csv_path
            (
                result.case,
                result.fields_physical,
                result.blade_mask,
                result.phi,
                result.metrics,
                result.blade_path,
                result.source_metadata,
                _,
            ) = _postprocess_csv(
                resolved,
                csv_path,
                documents,
                near_wall_distance_m,
                grid_size,
                interpolation_chunk_size,
                diagnostics,
            )

        if result.metrics is not None:
            if result.metrics.mass_imbalance_ratio > 0.05:
                _diag(diagnostics, "warning", "MASS_IMBALANCE", "进出口质量不平衡超过 5%。")
            if result.metrics.flow_mismatch_ratio > 0.10:
                _diag(diagnostics, "warning", "FLOW_MISMATCH", "预测流量与工况流量偏差超过 10%。")
            if not math.isfinite(result.metrics.hydraulic_efficiency):
                _diag(diagnostics, "warning", "EFFICIENCY_NONFINITE", "水力效率不是有限值，通常由近零转矩导致。")
            elif not 0.0 <= result.metrics.hydraulic_efficiency <= 1.0:
                _diag(
                    diagnostics,
                    "warning",
                    "EFFICIENCY_OUT_OF_RANGE",
                    "水力效率不在 0–100% 物理区间，请结合收敛性与 CFD 复核。",
                )
    except Exception as exc:
        _diag(diagnostics, "error", "POSTPROCESS_FAILED", f"数据集后处理失败：{type(exc).__name__}: {exc}", resolved)
    return result


def _canonical_duplicate_key(path: str | Path) -> str:
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        resolved = Path(path).expanduser().absolute()
    return str(resolved).casefold()


def postprocess_dataset_folders(
    folders: Sequence[str | Path],
    *,
    near_wall_distance_m: float = 0.003,
    grid_size: int = 64,
    interpolation_chunk_size: int = 250_000,
) -> DatasetBatchPostprocessResult:
    """Post-process one to three folder records, preserving duplicates.

    Repeated canonical paths reuse the first computed arrays/metrics but remain
    separate records.  Their ``duplicate`` flag is true and
    ``duplicate_of_index`` points to the first occurrence (zero based).
    """

    values = list(folders)
    if not 1 <= len(values) <= 3:
        raise ValueError("Batch post-processing accepts one to three folder records.")
    results: list[DatasetPostprocessResult] = []
    first_by_path: dict[str, int] = {}
    for index, raw_path in enumerate(values):
        key = _canonical_duplicate_key(raw_path)
        if key not in first_by_path:
            first_by_path[key] = index
            results.append(
                postprocess_dataset_folder(
                    raw_path,
                    near_wall_distance_m=near_wall_distance_m,
                    grid_size=grid_size,
                    interpolation_chunk_size=interpolation_chunk_size,
                    record_index=index,
                )
            )
            continue

        first_index = first_by_path[key]
        original = results[first_index]
        duplicate_diagnostics = list(original.diagnostics)
        _diag(
            duplicate_diagnostics,
            "info",
            "DUPLICATE_PATH",
            f"该路径与第 {first_index + 1} 条记录重复；保留本条记录并复用首次计算结果。",
            raw_path,
        )
        duplicate = replace(
            original,
            record_index=index,
            requested_path=str(raw_path),
            duplicate=True,
            duplicate_of_index=first_index,
            diagnostics=duplicate_diagnostics,
        )
        results.append(duplicate)
    return DatasetBatchPostprocessResult(results=results, near_wall_distance_m=float(near_wall_distance_m))


__all__ = [
    "DatasetBatchPostprocessResult",
    "DatasetPostprocessResult",
    "PostprocessDiagnostic",
    "postprocess_dataset_folder",
    "postprocess_dataset_folders",
]
