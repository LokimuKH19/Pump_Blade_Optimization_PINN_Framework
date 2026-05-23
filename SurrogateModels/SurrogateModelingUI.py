# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "surrogate_ui_configs"
LOG_DIR = ROOT / "surrogate_ui_logs"
SURROGATE_SCRIPT = ROOT / "SurrogateModeling.py"
ARTIFACT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".json",
    ".pt",
    ".pth",
    ".csv",
    ".log",
    ".txt",
    ".npy",
    ".npz",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
ARTIFACT_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^\s\"'<>|]+?\.(?:png|jpg|jpeg|webp|json|pt|pth|csv|log|txt|npy|npz))",
    re.IGNORECASE,
)
OUTPUT_DIR_RE = re.compile(r"(?:输出目录|Output directory|Save directory)\s*[:：]\s*(?P<path>.+)", re.IGNORECASE)

THEME_CFG = {
    "dark": {
        "page_bg": "#0E1117",
        "panel_bg": "#161B22",
        "text": "#E6EDF3",
        "muted": "#8B949E",
        "accent": "#58A6FF",
        "border": "#30363D",
    },
    "light": {
        "page_bg": "#FFFFFF",
        "panel_bg": "#F6F8FA",
        "text": "#24292F",
        "muted": "#57606A",
        "accent": "#0969DA",
        "border": "#D0D7DE",
    },
}
CONVECTION_OPTIONS = ["upwind2", "upwind", "central"]
CONVECTION_ALIASES = {
    "central": "central",
    "centered": "central",
    "upwind": "upwind",
    "upwind1": "upwind",
    "first_order_upwind": "upwind",
    "1st_order_upwind": "upwind",
    "upwind2": "upwind2",
    "second_order_upwind": "upwind2",
    "2nd_order_upwind": "upwind2",
}


def normalize_convection_interpolation(value: Any, default: str = "upwind2") -> str:
    key = str(value if value is not None else default).strip().lower().replace("-", "_").replace(" ", "_")
    return CONVECTION_ALIASES.get(key, default)


def apply_theme(theme: str) -> None:
    cfg = THEME_CFG[theme]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {cfg["page_bg"]};
            color: {cfg["text"]};
        }}
        section[data-testid="stSidebar"] {{
            background: {cfg["panel_bg"]};
            border-right: 1px solid {cfg["border"]};
        }}
        div[data-testid="stMetricValue"] {{
            color: {cfg["accent"]};
        }}
        .surrogate-card {{
            padding: 1rem 1.1rem;
            border: 1px solid {cfg["border"]};
            border-radius: 8px;
            background: {cfg["panel_bg"]};
            color: {cfg["text"]};
        }}
        .surrogate-muted {{
            color: {cfg["muted"]};
            font-size: 0.9rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def discover_simulation_dirs() -> list[Path]:
    roots = [
        ROOT / "../BladeOptimizerLFR",
        ROOT,
        ROOT / "generated_flow_cases",
        ROOT / "generated_flow_cases_3d",
        ROOT / "generated_flow_cases_xyz",
    ]
    matches: list[Path] = []
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.exists():
            continue
        matches.extend(path for path in root.glob("*_SIMULATION") if path.is_dir())
        matches.extend(path.parent for path in root.rglob("blade_params.json"))
    unique = {str(path.resolve()): path.resolve() for path in matches}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)


def discover_csv_files(folders: list[str] | None = None) -> list[Path]:
    search_roots = [Path(folder) for folder in folders or [] if str(folder).strip()]
    if not search_roots:
        search_roots = [ROOT / "../BladeOptimizerLFR", ROOT]
    matches: list[Path] = []
    for raw_root in search_roots:
        root = raw_root.resolve()
        if root.is_file() and root.suffix.lower() == ".csv":
            matches.append(root)
        elif root.exists():
            matches.extend(path for path in root.glob("*.csv") if path.is_file())
            matches.extend(path for path in root.rglob("*.csv") if path.is_file())
    unique = {str(path.resolve()): path.resolve() for path in matches}
    return sorted(unique.values(), key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)


def find_latest_checkpoint(root: Path) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob("surrogate_checkpoint.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def default_config() -> dict[str, Any]:
    simulation_candidates = discover_simulation_dirs()
    default_sim = (
        str(simulation_candidates[0])
        if simulation_candidates
        else str((ROOT / "../BladeOptimizerLFR/CQ_20260514_115826_SIMULATION").resolve())
    )
    return {
        "workflow_action": "train",
        "training_mode": "mixed",
        "checkpoint_to_load": None,
        "simulation_folders": [default_sim],
        "cfd_csv_files": [],
        "output_root": "surrogate_formal",
        "rpm": -210.0,
        "physics_config": {
            "mu": 0.006,
            "rho": 10650.0,
            "qv": 0.025,
            "g": 9.8,
        },
        "data_config": {
            "n": 64,
            "theta_sector_index": 0,
            "interpolation_chunk_size": 250_000,
        },
        "model_config": {
            "operator_variant": "hf_cfno",
            "input_mode": "both",
            "modes": 8,
            "high_modes": 16,
            "width": 16,
            "depth": 6,
            "z_padding": 8,
            "fourier_feature_bands": [1, 2, 4, 8],
            "hf_high_gate_init": -1.0,
            "hf_use_local_highpass": True,
            "pressure_smoothing": 0.0,
            "pressure_supervision_mode": "value",
            "pressure_reference_mode": "origin",
            "pressure_data_reference": "absolute",
        },
        "training_config": {
            "epochs": 500,
            "print_interval": 1,
            "checkpoint_interval": 50,
            "prefer_existing_run_checkpoint": True,
            "batch_size": 1,
            "lr": 1e-3,
            "pressure_highpass_weight": 1e8,
            "pressure_highpass_normalized": True,
            "physics_discretization": "fvm_rhie_chow",
            "rhie_chow_strength": 0.35,
            "momentum_diagonal_floor": 1.0,
            "convection_interpolation": "upwind2",
            "use_kkt_projection": False,
            "kkt_projection_iters": 24,
            "kkt_projection_strength": 0.35,
            "learn_ibm_params": True,
            "ibm_c_range": [0.3, 3.0],
            "ibm_epsilon_range": [0.001, 0.05],
            "micro_batch_size": None,
            "slice_batch_size": None,
            "auto_cuda_batching": True,
            "cuda_memory_fraction": 0.80,
            "use_activation_checkpointing": True,
            "device": "cuda",
        },
        "post_config": {
            "spans": [0.4, 0.6],
            "show_matplotlib": False,
            "show_pyvista_window": False,
            "plot_3d": False,
            "history_plot_mode": "all",
            "passages_to_plot_3d": None,
            "cfd_pressure_reference": "absolute",
            "interpolation_chunk_size": 250_000,
        },
        "run_fine_grid_deploy": False,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(config: dict[str, Any], name: str | None = None) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = name.strip() if name else ""
    if not stem:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"surrogate_{config['workflow_action']}_{config['training_mode']}_{stamp}"
    if not stem.endswith(".json"):
        stem += ".json"
    path = CONFIG_DIR / stem
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_float_list(text: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values or default


def parse_optional_int(value: int, enabled: bool) -> int | None:
    return int(value) if enabled else None


def parse_optional_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def launch_run(config_path: Path) -> tuple[int, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{config_path.stem}_{stamp}.log"
    command = [sys.executable, "-u", str(SURROGATE_SCRIPT), "--main-config", str(config_path)]
    log_file = log_path.open("w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    finally:
        log_file.close()
    return proc.pid, log_path


def read_log(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        return ""
    raw = path.read_bytes()
    text = decode_text_bytes(raw)
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return text[-max_chars:]
    return text


def decode_text_bytes(raw: bytes) -> str:
    if not raw:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp936"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_tail(path: Path, max_chars: int = 12000) -> str:
    return read_log(path, max_chars=max_chars)


def discover_log_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return sorted(LOG_DIR.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)


def normalize_artifact_path(raw_path: str) -> Path | None:
    cleaned = raw_path.strip().strip("\"'`[]<>")
    cleaned = cleaned.rstrip(").,;，。")
    if not cleaned:
        return None
    path = Path(cleaned)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def extract_output_dirs(log_text: str) -> list[Path]:
    output_dirs: dict[str, Path] = {}
    for line in log_text.splitlines():
        match = OUTPUT_DIR_RE.search(line)
        if not match:
            continue
        path = normalize_artifact_path(match.group("path"))
        if path is not None:
            output_dirs[str(path)] = path
    return list(output_dirs.values())


def discover_artifacts_from_dirs(output_dirs: list[Path], max_files: int = 500) -> list[Path]:
    artifacts: dict[str, Path] = {}
    for output_dir in output_dirs:
        if not output_dir.exists() or not output_dir.is_dir():
            continue
        for path in output_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ARTIFACT_SUFFIXES:
                continue
            artifacts[str(path.resolve())] = path.resolve()
            if len(artifacts) >= max_files:
                return list(artifacts.values())
    return list(artifacts.values())


def extract_artifact_paths(log_text: str) -> list[Path]:
    artifacts: dict[str, Path] = {}
    for match in ARTIFACT_PATH_RE.finditer(log_text):
        path = normalize_artifact_path(match.group("path"))
        if path is not None:
            artifacts[str(path)] = path
    for path in discover_artifacts_from_dirs(extract_output_dirs(log_text)):
        artifacts[str(path)] = path
    return sorted(
        artifacts.values(),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix == ".json":
        return "json"
    if suffix in {".pt", ".pth"}:
        return "checkpoint"
    if suffix == ".csv":
        return "csv"
    if suffix == ".log":
        return "log"
    return suffix.lstrip(".") or "file"


def format_file_size(path: Path) -> str:
    if not path.exists():
        return ""
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def artifact_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        exists = path.exists()
        modified = ""
        if exists:
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        rows.append(
            {
                "type": artifact_kind(path),
                "exists": "yes" if exists else "missing",
                "size": format_file_size(path),
                "modified": modified,
                "path": str(path),
            }
        )
    return rows


def rerun_streamlit() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is not None:
        rerun()


def sidebar_config() -> dict[str, Any]:
    st.sidebar.header("Surrogate Workflow")
    theme = st.sidebar.radio(
        "UI Theme",
        ["dark", "light"],
        index=0 if st.session_state.get("theme_mode", "dark") == "dark" else 1,
        key="theme_mode",
    )
    apply_theme(theme)

    existing_configs = sorted(CONFIG_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    selected_config = st.sidebar.selectbox(
        "加载已有配置",
        ["<默认配置>"] + [path.name for path in existing_configs],
    )
    if selected_config != "<默认配置>":
        base = load_config(CONFIG_DIR / selected_config)
    else:
        base = default_config()

    action = st.sidebar.radio(
        "工作流",
        ["train", "resume_train", "deploy"],
        index=["train", "resume_train", "deploy"].index(base.get("workflow_action", "train")),
        captions=["从头训练", "从 checkpoint 续训", "只部署/后处理"],
    )
    training_mode = st.sidebar.radio(
        "训练/样本模式",
        ["mixed", "data_only", "physics_only"],
        index=["mixed", "data_only", "physics_only"].index(base.get("training_mode", "mixed")),
        captions=["CFD 数据 + 物理", "只监督 CFD 数据", "只用叶片几何和物理"],
    )
    output_root = st.sidebar.text_input("输出根目录", value=str(base.get("output_root", "surrogate_formal")))

    checkpoint_default = base.get("checkpoint_to_load")
    if checkpoint_default == "latest" or (action in {"resume_train", "deploy"} and checkpoint_default is None):
        checkpoint_mode = "latest"
    elif checkpoint_default is None:
        checkpoint_mode = "none"
    else:
        checkpoint_mode = "path"
    checkpoint_mode = st.sidebar.radio("Checkpoint", ["latest", "path", "none"], index=["latest", "path", "none"].index(checkpoint_mode))
    checkpoint_root = (ROOT / output_root).resolve()
    discovered_checkpoints = sorted(
        checkpoint_root.rglob("surrogate_checkpoint.pt") if checkpoint_root.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    checkpoint_choices = ["<manual>"] + [str(path) for path in discovered_checkpoints]
    default_choice = str(checkpoint_default) if checkpoint_default not in {None, "latest"} else "<manual>"
    selected_checkpoint = st.sidebar.selectbox(
        "Discovered models",
        checkpoint_choices,
        index=checkpoint_choices.index(default_choice) if default_choice in checkpoint_choices else 0,
        disabled=checkpoint_mode != "path",
    )
    checkpoint_text_default = (
        selected_checkpoint
        if selected_checkpoint != "<manual>"
        else ("" if checkpoint_default in {None, "latest"} else str(checkpoint_default))
    )
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint 路径",
        value=checkpoint_text_default,
        disabled=checkpoint_mode != "path",
    )
    checkpoint_to_load: str | None
    if checkpoint_mode == "latest":
        checkpoint_to_load = "latest"
    elif checkpoint_mode == "path":
        checkpoint_to_load = parse_optional_text(checkpoint_path)
    else:
        checkpoint_to_load = None

    latest = find_latest_checkpoint((ROOT / output_root).resolve())
    if latest is not None:
        st.sidebar.caption(f"Latest: {latest}")

    base["workflow_action"] = action
    base["training_mode"] = training_mode
    base["checkpoint_to_load"] = checkpoint_to_load
    base["output_root"] = output_root
    return base


def render_simulation_picker(config: dict[str, Any]) -> None:
    candidates = discover_simulation_dirs()
    current = "\n".join(str(path) for path in config.get("simulation_folders", []))
    st.subheader("Simulation Cases")
    st.caption("每行一个仿真目录。mixed/data_only 会读取 CFD 字段；physics_only 只需要目录里有 blade_params.json。")
    text = st.text_area("仿真目录列表", value=current, height=120)
    selected = st.multiselect(
        "从已发现目录中添加",
        [str(path) for path in candidates],
        default=[],
    )
    folders = [line.strip() for line in text.splitlines() if line.strip()]
    folders.extend(path for path in selected if path not in folders)
    config["simulation_folders"] = folders

    st.subheader("CFD CSV For Deploy / Comparison")
    st.caption("可留空：程序会默认在 blade_params.json 同路径的仿真目录中寻找唯一 CSV。需要指定时每行一个 CSV，顺序对应上面的目录。")
    current_csv = "\n".join(str(path) for path in config.get("cfd_csv_files", []))
    csv_text = st.text_area("CFD CSV 列表", value=current_csv, height=100)
    csv_candidates = discover_csv_files(folders)
    selected_csv = st.multiselect(
        "从目录中发现的 CSV 添加",
        [str(path) for path in csv_candidates],
        default=[],
    )
    csv_files = [line.strip() for line in csv_text.splitlines() if line.strip()]
    csv_files.extend(path for path in selected_csv if path not in csv_files)
    config["cfd_csv_files"] = csv_files


def render_physics_tab(config: dict[str, Any]) -> None:
    physics = config["physics_config"]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        config["rpm"] = st.number_input("RPM", value=float(config.get("rpm", -210.0)), step=5.0, format="%.6f")
    with col2:
        physics["mu"] = st.number_input("mu", value=float(physics.get("mu", 0.006)), step=0.001, format="%.8f")
    with col3:
        physics["rho"] = st.number_input("rho", value=float(physics.get("rho", 10650.0)), step=100.0, format="%.6f")
    with col4:
        physics["qv"] = st.number_input("qv total", value=float(physics.get("qv", 0.025)), step=0.005, format="%.8f")
    with col5:
        physics["g"] = st.number_input("g", value=float(physics.get("g", 9.8)), step=0.1, format="%.6f")


def render_model_tab(config: dict[str, Any]) -> None:
    data = config["data_config"]
    model = config["model_config"]

    st.subheader("Grid")
    col1, col2, col3 = st.columns(3)
    with col1:
        data["n"] = st.number_input("Grid n", min_value=8, max_value=512, value=int(data.get("n", 64)), step=8)
    with col2:
        data["theta_sector_index"] = st.number_input(
            "Theta sector",
            min_value=0,
            max_value=64,
            value=int(data.get("theta_sector_index", 0)),
            step=1,
        )
    with col3:
        data["interpolation_chunk_size"] = st.number_input(
            "Interpolation chunk",
            min_value=10_000,
            max_value=2_000_000,
            value=int(data.get("interpolation_chunk_size", 250_000)),
            step=10_000,
        )

    st.subheader("Operator")
    variants = [
        "fno",
        "hf_fno",
        "cno",
        "wno",
        "cfno",
        "hf_cfno",
        "fno3d",
        "hf_fno3d",
        "cno3d",
        "wno3d",
        "cfno3d",
        "hf_cfno3d",
    ]
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        model["operator_variant"] = st.selectbox(
            "Operator",
            variants,
            index=variants.index(model.get("operator_variant", "hf_cfno")) if model.get("operator_variant", "hf_cfno") in variants else variants.index("hf_cfno"),
        )
    with col2:
        model["input_mode"] = st.selectbox("Input", ["both", "mask", "phi"], index=["both", "mask", "phi"].index(model.get("input_mode", "both")))
    with col3:
        model["modes"] = st.number_input("Modes", min_value=1, max_value=64, value=int(model.get("modes", 8)), step=1)
    with col4:
        high_enabled = st.checkbox("High modes", value=model.get("high_modes", None) is not None)
        model["high_modes"] = parse_optional_int(
            st.number_input("High", min_value=1, max_value=128, value=int(model.get("high_modes") or 16), step=1),
            high_enabled,
        )
    with col5:
        model["z_padding"] = st.number_input("Z padding", min_value=0, max_value=64, value=int(model.get("z_padding", 8)), step=1)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model["width"] = st.number_input("Width", min_value=2, max_value=256, value=int(model.get("width", 16)), step=2)
    with col2:
        model["depth"] = st.number_input("Depth", min_value=1, max_value=32, value=int(model.get("depth", 6)), step=1)
    with col3:
        bands = st.text_input("Fourier bands", value=",".join(str(v) for v in model.get("fourier_feature_bands", [1, 2, 4, 8])))
        model["fourier_feature_bands"] = [int(v) for v in parse_float_list(bands, [1, 2, 4, 8])]
    with col4:
        model["hf_high_gate_init"] = st.number_input(
            "HF gate init",
            value=float(model.get("hf_high_gate_init", -1.0)),
            step=0.1,
            format="%.4f",
        )
    model["hf_use_local_highpass"] = st.checkbox("HF local highpass", value=bool(model.get("hf_use_local_highpass", True)))

    st.subheader("Pressure")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model["pressure_supervision_mode"] = st.selectbox(
            "Supervision",
            ["value", "gradient", "none"],
            index=["value", "gradient", "none"].index(model.get("pressure_supervision_mode", "value")),
        )
    with col2:
        model["pressure_reference_mode"] = st.selectbox(
            "Output reference",
            ["origin", "none"],
            index=["origin", "none"].index(model.get("pressure_reference_mode", "origin")),
        )
    with col3:
        model["pressure_data_reference"] = st.selectbox(
            "Data reference",
            ["absolute", "training_origin"],
            index=["absolute", "training_origin"].index(model.get("pressure_data_reference", "absolute")),
        )
    with col4:
        model["pressure_smoothing"] = st.number_input(
            "Smoothing",
            min_value=0.0,
            max_value=1.0,
            value=float(model.get("pressure_smoothing", 0.0)),
            step=0.05,
        )


def render_training_tab(config: dict[str, Any]) -> None:
    training = config["training_config"]
    action = config["workflow_action"]

    st.subheader("Training Loop")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        training["epochs"] = st.number_input(
            "Epochs",
            min_value=0,
            max_value=200_000,
            value=0 if action == "deploy" else int(training.get("epochs", 500)),
            step=10,
            disabled=action == "deploy",
        )
    with col2:
        training["print_interval"] = st.number_input("Print interval", min_value=1, max_value=10_000, value=int(training.get("print_interval", 1)), step=1)
    with col3:
        training["checkpoint_interval"] = st.number_input(
            "Checkpoint interval",
            min_value=0,
            max_value=50_000,
            value=int(training.get("checkpoint_interval", 50)),
            step=10,
        )
    with col4:
        training["batch_size"] = st.number_input("Batch size", min_value=1, max_value=128, value=int(training.get("batch_size", 1)), step=1)
    with col5:
        training["lr"] = st.number_input("Learning rate", min_value=1e-8, max_value=1.0, value=float(training.get("lr", 1e-3)), step=1e-4, format="%.8f")

    st.subheader("Physics Loss")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        training["physics_discretization"] = st.selectbox(
            "Discretization",
            ["fvm_rhie_chow", "centered"],
            index=["fvm_rhie_chow", "centered"].index(training.get("physics_discretization", "fvm_rhie_chow")),
        )
    with col2:
        convection_value = normalize_convection_interpolation(training.get("convection_interpolation", "upwind2"))
        training["convection_interpolation"] = st.selectbox(
            "Convection",
            CONVECTION_OPTIONS,
            index=CONVECTION_OPTIONS.index(convection_value),
            format_func=lambda value: {
                "upwind2": "upwind2 (2nd order)",
                "upwind": "upwind (1st order)",
                "central": "central",
            }.get(value, value),
        )
    with col3:
        training["rhie_chow_strength"] = st.number_input("Rhie-Chow", min_value=0.0, max_value=5.0, value=float(training.get("rhie_chow_strength", 0.35)), step=0.05)
    with col4:
        training["momentum_diagonal_floor"] = st.number_input("Diag floor", min_value=1e-8, max_value=100.0, value=float(training.get("momentum_diagonal_floor", 1.0)), step=0.1)
    with col5:
        training["pressure_highpass_weight"] = st.number_input(
            "P highpass weight",
            min_value=0.0,
            max_value=1e12,
            value=float(training.get("pressure_highpass_weight", 1e8)),
            step=1e6,
            format="%.6g",
        )
    training["pressure_highpass_normalized"] = st.checkbox("Normalize pressure highpass", value=bool(training.get("pressure_highpass_normalized", True)))

    st.subheader("KKT / IBM / Memory")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        training["use_kkt_projection"] = st.checkbox("KKT projection", value=bool(training.get("use_kkt_projection", False)))
        training["kkt_projection_iters"] = st.number_input("KKT iters", min_value=1, max_value=512, value=int(training.get("kkt_projection_iters", 24)), step=1)
    with col2:
        training["kkt_projection_strength"] = st.number_input("KKT strength", min_value=0.0, max_value=5.0, value=float(training.get("kkt_projection_strength", 0.35)), step=0.05)
        training["learn_ibm_params"] = st.checkbox("Learn IBM params", value=bool(training.get("learn_ibm_params", True)))
    with col3:
        c_range = st.text_input("IBM C range", value=",".join(str(v) for v in training.get("ibm_c_range", [0.3, 3.0])))
        training["ibm_c_range"] = parse_float_list(c_range, [0.3, 3.0])[:2]
        eps_range = st.text_input("IBM epsilon range", value=",".join(str(v) for v in training.get("ibm_epsilon_range", [0.001, 0.05])))
        training["ibm_epsilon_range"] = parse_float_list(eps_range, [0.001, 0.05])[:2]
    with col4:
        training["device"] = st.selectbox("Device", ["cuda", "cpu"], index=["cuda", "cpu"].index(training.get("device", "cuda")))
        training["cuda_memory_fraction"] = st.slider("CUDA memory fraction", min_value=0.05, max_value=0.95, value=float(training.get("cuda_memory_fraction", 0.80)), step=0.05)
    training["auto_cuda_batching"] = st.checkbox("Auto CUDA batching", value=bool(training.get("auto_cuda_batching", True)))
    training["use_activation_checkpointing"] = st.checkbox("Activation checkpointing", value=bool(training.get("use_activation_checkpointing", True)))


def render_post_tab(config: dict[str, Any]) -> None:
    post = config["post_config"]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        spans = st.text_input("Span list", value=",".join(str(v) for v in post.get("spans", [0.4, 0.6])))
        post["spans"] = parse_float_list(spans, [0.4, 0.6])
    with col2:
        history_options = ["all", "scaled_residual", "scaled_loss", "raw", "fluent_scaled", "auto"]
        current_history_mode = str(post.get("history_plot_mode", "all")).lower()
        if current_history_mode == "scaled":
            current_history_mode = "scaled_loss"
        if current_history_mode == "both":
            current_history_mode = "all"
        if current_history_mode not in history_options:
            current_history_mode = "all"
        post["history_plot_mode"] = st.selectbox(
            "History curves",
            history_options,
            index=history_options.index(current_history_mode),
        )
    with col3:
        post["cfd_pressure_reference"] = st.selectbox(
            "CFD pressure ref",
            ["absolute", "training_origin"],
            index=["absolute", "training_origin"].index(post.get("cfd_pressure_reference", "absolute")),
        )
    with col4:
        post["interpolation_chunk_size"] = st.number_input(
            "Post chunk",
            min_value=10_000,
            max_value=2_000_000,
            value=int(post.get("interpolation_chunk_size", config["data_config"].get("interpolation_chunk_size", 250_000))),
            step=10_000,
        )

    post["show_matplotlib"] = st.checkbox("Show Matplotlib windows", value=bool(post.get("show_matplotlib", False)))
    post["show_pyvista_window"] = st.checkbox("Show PyVista window", value=bool(post.get("show_pyvista_window", False)))
    post["plot_3d"] = st.checkbox("Generate 3D streamlines", value=bool(post.get("plot_3d", False)))
    passages_enabled = st.checkbox("Limit copied passages", value=post.get("passages_to_plot_3d") is not None)
    post["passages_to_plot_3d"] = parse_optional_int(
        st.number_input("Passages to plot", min_value=1, max_value=128, value=int(post.get("passages_to_plot_3d") or 1), step=1),
        passages_enabled,
    )
    config["run_fine_grid_deploy"] = st.checkbox("Run optional fine-grid deploy", value=bool(config.get("run_fine_grid_deploy", False)))


def main() -> None:
    st.set_page_config(page_title="Surrogate Modeling Workbench", layout="wide")
    config = sidebar_config()

    st.title("Surrogate Modeling Workbench")
    st.markdown(
        '<div class="surrogate-muted">Configure, launch, and monitor SurrogateModeling.py for training, resume-training, and deployment.</div>',
        unsafe_allow_html=True,
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Workflow", config["workflow_action"])
    summary_cols[1].metric("Mode", config["training_mode"])
    summary_cols[2].metric("Grid n", config["data_config"].get("n", 64))
    summary_cols[3].metric("Operator", config["model_config"].get("operator_variant", "hf_cfno"))

    tabs = st.tabs(["Cases", "Physics", "Model", "Training", "Post", "Launch"])
    with tabs[0]:
        render_simulation_picker(config)
    with tabs[1]:
        render_physics_tab(config)
    with tabs[2]:
        render_model_tab(config)
    with tabs[3]:
        render_training_tab(config)
    with tabs[4]:
        render_post_tab(config)
    with tabs[5]:
        st.subheader("Config Preview")
        config_name = st.text_input("配置文件名", value="")
        st.json(config)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("保存配置", type="secondary", use_container_width=True):
                path = save_config(config, config_name)
                st.session_state["last_config_path"] = str(path)
                st.success(f"已保存: {path}")
        with col2:
            if st.button("保存并启动", type="primary", use_container_width=True):
                path = save_config(config, config_name)
                pid, log_path = launch_run(path)
                st.session_state["last_config_path"] = str(path)
                st.session_state["last_log_path"] = str(log_path)
                st.session_state["last_pid"] = int(pid)
                st.success(f"已启动 PID={pid}")
        with col3:
            preview_path = st.session_state.get("last_config_path", "<保存配置后生成>")
            st.code(f"{sys.executable} -u {SURROGATE_SCRIPT} --main-config {preview_path}", language="powershell")

        st.subheader("Run Output / 完整日志")
        log_files = discover_log_files()
        last_log = st.session_state.get("last_log_path")
        if log_files:
            log_options = [str(path) for path in log_files]
            default_log_index = log_options.index(last_log) if last_log in log_options else 0
            selected_log = st.selectbox("运行日志", log_options, index=default_log_index)
            st.session_state["last_log_path"] = selected_log
            log_path = Path(selected_log)
            log_text = read_log(log_path)

            log_col1, log_col2, log_col3 = st.columns([1, 1, 2])
            with log_col1:
                if st.button("刷新日志", use_container_width=True):
                    rerun_streamlit()
            with log_col2:
                full_log = st.checkbox("显示完整 print 输出", value=True)
            with log_col3:
                st.download_button(
                    "下载日志",
                    data=log_text,
                    file_name=log_path.name,
                    mime="text/plain",
                    use_container_width=True,
                )

            if full_log:
                shown_log = log_text
                log_label = "完整 print 输出"
            else:
                max_chars = st.slider("尾部字符数", min_value=2_000, max_value=80_000, value=12_000, step=2_000)
                shown_log = read_tail(log_path, max_chars=max_chars)
                log_label = "print 输出尾部"
            st.caption(f"Log: {log_path}")
            st.text_area(log_label, value=shown_log, height=460)

            st.subheader("Output Artifacts / 输出产物位置")
            artifact_paths = extract_artifact_paths(log_text)
            rows = artifact_rows(artifact_paths)
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
                image_paths = [path for path in artifact_paths if path.suffix.lower() in IMAGE_SUFFIXES and path.exists()]
                if image_paths:
                    st.code("\n".join(str(path) for path in image_paths), language="text")
                    if st.checkbox("预览输出图片", value=True):
                        for image_path in image_paths[:12]:
                            st.image(str(image_path), caption=str(image_path), use_container_width=True)
                        if len(image_paths) > 12:
                            st.caption(f"还有 {len(image_paths) - 12} 张图片未预览，但路径已在上方表格中列出。")
            else:
                st.info("当前日志里还没有发现输出图片或产物路径。")
        else:
            st.info("还没有运行日志。点击“保存并启动”后，这里会显示完整 print 输出和产物位置。")


if __name__ == "__main__":
    main()
