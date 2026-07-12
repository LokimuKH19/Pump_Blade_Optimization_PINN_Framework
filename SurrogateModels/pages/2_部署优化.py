from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SurrogateDeployment import (  # noqa: E402
    DeploymentEngine,
    DeploymentPrediction,
    EffectivePumpMetricsEngine,
    OptimizationResult,
    apply_blade_parameter_rows,
    blade_parameter_rows,
    build_training_envelope_variables,
    candidate_rows,
    checkpoint_training_folders,
    discover_model_checkpoints,
    effective_pump_metrics,
    inspect_checkpoint,
    load_blade_parameters,
    optimize_blade_design,
    prediction_to_npz_bytes,
    resolve_blade_parameters_path,
    shape_change_summary,
)


st.set_page_config(page_title="叶片代理模型部署与优化", page_icon="⚙️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: .55rem; padding: .65rem .8rem;}
    div[data-testid="stAlert"] {border-radius: .55rem;}
    .deploy-kicker {letter-spacing:.08em; text-transform:uppercase; font-size:.75rem; opacity:.65; margin-bottom:.2rem;}
    .deploy-note {font-size:.88rem; opacity:.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


SUPPORTED_VARIANTS = {"fno", "hf_fno", "cfno", "hf_cfno", "fno3d"}


@st.cache_data(show_spinner=False, ttl=120)
def preflight_catalog() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    ready: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for entry in discover_model_checkpoints(ROOT / "surrogate_formal"):
        try:
            if not zipfile.is_zipfile(entry.path):
                raise ValueError("checkpoint ZIP container is incomplete or corrupted")
            variant = str(entry.operator_variant)
            if variant and variant not in SUPPORTED_VARIANTS:
                raise ValueError(f"unsupported operator variant: {variant or 'unknown'}")
            ready.append(
                {
                    "path": str(entry.path),
                    "name": entry.name,
                    "label": f"{entry.label} · ZIP-ready",
                    "size_mb": entry.size_bytes / 1024**2,
                }
            )
        except Exception as exc:
            rejected.append({"name": entry.name, "path": str(entry.path), "reason": str(exc)})
    return ready, rejected


@st.cache_data(show_spinner=False)
def cached_metadata(checkpoint_path: str) -> dict[str, Any]:
    return inspect_checkpoint(checkpoint_path)


@st.cache_data(show_spinner=False)
def cached_context(
    checkpoint_path: str,
    case_index: int,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    metadata = cached_metadata(checkpoint_path)
    summaries = metadata["train_case_summaries"]
    if not 0 <= case_index < len(summaries):
        raise IndexError("Stored case index is outside checkpoint metadata.")
    case = dict(summaries[case_index])
    raw_blade_path = case.get("blade_params")
    if raw_blade_path is None:
        raise ValueError("The selected checkpoint case has no blade_params reference.")
    blade_path = resolve_blade_parameters_path(raw_blade_path, checkpoint_path)
    parameters = load_blade_parameters(blade_path)
    case["blade_params"] = parameters
    return metadata, case, str(blade_path), parameters


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_engine(checkpoint_path: str, case_index: int, device: str) -> DeploymentEngine:
    return DeploymentEngine(checkpoint_path, case_index=case_index, device=device)


def design_signature(checkpoint_path: str, case_index: int, parameters: dict[str, Any], d_m: float) -> str:
    payload = json.dumps(
        {"checkpoint": checkpoint_path, "case": case_index, "parameters": parameters, "d": d_m},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_number(value: float, unit: str = "", digits: int = 3) -> str:
    if not math.isfinite(float(value)):
        return "—"
    return f"{value:,.{digits}f}{unit}"


def render_locked_condition(case: dict[str, Any]) -> None:
    rpm = float(case["omega"]) * 60.0 / (2.0 * np.pi)
    first = st.columns(6)
    first[0].metric("网格", f"{int(case['n'])}³")
    first[1].metric("转速", f"{rpm:.1f} rpm")
    first[2].metric("体积流量", f"{float(case['qv']):.4f} m³/s")
    first[3].metric("密度", f"{float(case['rho']):.1f} kg/m³")
    first[4].metric("动力黏度", f"{float(case['mu']):.4g} Pa·s")
    first[5].metric("重力", f"{float(case.get('g', 9.8)):.3g} m/s²")
    second = st.columns(6)
    second[0].metric("轮毂半径", f"{float(case['rh'])*1000:.2f} mm")
    second[1].metric("轮缘半径", f"{float(case['rs'])*1000:.2f} mm")
    second[2].metric("流道高度", f"{float(case['h'])*1000:.2f} mm")
    second[3].metric("叶片数", str(int(case["n_blade"])))
    second[4].metric("轴向起点 z₀", f"{float(case.get('z0', 0.0))*1000:.2f} mm")
    second[5].metric("参考系", "绝对" if case.get("absolute_frame", True) else "旋转")


def render_performance(prediction: DeploymentPrediction, label: str) -> None:
    metrics = effective_pump_metrics(prediction.metrics)
    st.subheader(label)
    top = st.columns(6)
    top[0].metric("有效扬程", format_number(metrics.total_head_m, " m"))
    top[1].metric("静压差", format_number(metrics.static_pressure_rise_pa / 1000.0, " kPa", 2))
    top[2].metric("水力效率", "—" if not math.isfinite(metrics.hydraulic_efficiency) else f"{metrics.hydraulic_efficiency:.2%}")
    top[3].metric("混合特征流速", format_number(metrics.feature_velocity_m_s, " m/s"))
    top[4].metric("驱动转矩估算", format_number(metrics.driving_torque_n_m, " N·m"))
    top[5].metric("几何信赖", metrics.geometry_trust_label)

    detail = st.columns(6)
    detail[0].metric("近壁平均速度", format_number(metrics.relative_velocity_mean_m_s, " m/s"))
    detail[1].metric("近壁 P95", format_number(metrics.relative_velocity_p95_m_s, " m/s"))
    detail[2].metric("近壁最大速度", format_number(metrics.relative_velocity_max_m_s, " m/s"))
    detail[3].metric("轴输入功率", format_number(metrics.shaft_input_power_w / 1000.0, " kW"))
    detail[4].metric("水力输出功率", format_number(metrics.hydraulic_output_power_w / 1000.0, " kW"))
    detail[5].metric("预测总流量", format_number(metrics.predicted_flow_rate_m3_s, " m³/s", 4))

    head_parts = st.columns(4)
    head_parts[0].metric("压力水头", format_number(metrics.pressure_head_m, " m"))
    head_parts[1].metric("速度水头", format_number(metrics.velocity_head_m, " m"))
    head_parts[2].metric("高度水头", format_number(metrics.elevation_head_m, " m"))
    head_parts[3].metric("进出口质量失衡", f"{metrics.mass_imbalance_ratio:.2%}")

    if metrics.geometry_trust_label == "严重 OOD":
        st.error("当前几何已越出局部信赖域；此结果不能标记为可靠最优。")
    elif metrics.geometry_trust_label == "基准/网格不可分辨" and metrics.geometry_phi_relative_l2 > 0:
        st.warning("参数发生了变化，但当前 64³ 几何通道几乎无法分辨该扰动。")
    if metrics.notes:
        for note in metrics.notes:
            st.warning(note)

    st.caption(
        f"近壁 d={metrics.near_wall_distance_m*1000:.2f} mm，采样 {metrics.near_wall_cell_count:,} 点；"
        f"mask IoU={metrics.geometry_mask_iou:.5f}，体素变化={metrics.geometry_volume_change_ratio:.2%}，"
        f"流量工况偏差={metrics.flow_mismatch_ratio:.2%}。"
    )

    left, right = st.columns([1.2, 1])
    with left:
        field_options = {
            "相对叶面速度 |Vrel| [m/s]": (prediction.relative_speed_m_s, "m/s", 1.0),
            "径向速度 UR [m/s]": (prediction.fields_physical["UR"], "m/s", 1.0),
            "周向速度 UT [m/s]": (prediction.fields_physical["UT"], "m/s", 1.0),
            "轴向速度 UZ [m/s]": (prediction.fields_physical["UZ"], "m/s", 1.0),
            "压力 P [kPa]": (prediction.fields_physical["P"], "kPa", 1e-3),
        }
        field_name = st.selectbox("物理场", list(field_options), key=f"field_{prediction.checkpoint_path}_{label}")
        span = st.slider("径向截面 span", 0.0, 1.0, 0.5, 0.05, key=f"span_{prediction.checkpoint_path}_{label}")
        field, unit, scale = field_options[field_name]
        radial_index = int(round(span * (field.shape[0] - 1)))
        radius = prediction.coordinates.r_m[radial_index]
        arc_mm = radius * prediction.coordinates.theta_rad * 1000.0
        z_mm = prediction.coordinates.z_m * 1000.0
        values = np.asarray(field[radial_index], dtype=float).T * scale
        blade = prediction.blade_mask[radial_index].T
        values = np.ma.masked_where(blade, values)
        arc_grid_mm, z_grid_mm = np.meshgrid(arc_mm, z_mm)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
        fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
        ax.set_facecolor("#e6e6e6")
        image = ax.pcolormesh(arc_grid_mm, z_grid_mm, values, shading="auto", cmap=cmap)
        ax.contour(arc_grid_mm, z_grid_mm, blade.astype(float), levels=[0.5], colors="black", linewidths=1.0)
        ax.set_xlabel("周向弧长 rθ [mm]")
        ax.set_ylabel("物理轴向坐标 z [mm]")
        ax.set_title(f"span={span:.2f}, r={radius*1000:.2f} mm")
        fig.colorbar(image, ax=ax, label=unit)
        st.pyplot(fig, width="stretch")
        st.caption("灰色区域来自 BladeImport 的二值浸没边界固体掩膜；该区域统一为 NaN，不参与色标与流场后处理。")
        plt.close(fig)
    with right:
        st.markdown("**物理空间映射**")
        st.latex(r"r=r_h+R(r_s-r_h),\quad \theta=\Theta\,2\pi/N,\quad z=z_0+Zh")
        st.latex(r"x=r\cos\theta,\quad y=r\sin\theta")
        st.markdown("**总扬程（含重力）**")
        st.latex(r"H=\Delta\left(\frac{p}{\rho g}+\frac{|V|^2}{2g}+z\right)")
        st.markdown("**功率与效率**")
        st.latex(r"P_h=\rho gQH,\quad P_{shaft}=|T\omega|,\quad \eta_h=P_h/P_{shaft}")
        if prediction.compatibility_migration:
            st.info(f"兼容迁移：{prediction.compatibility_migration}（原 checkpoint 未修改）")

    export_key = f"npz_{design_signature(prediction.checkpoint_path, 0, prediction.blade_parameters, metrics.near_wall_distance_m)}"
    if export_key not in st.session_state:
        st.session_state[export_key] = prediction_to_npz_bytes(prediction)
    downloads = st.columns(3)
    downloads[0].download_button(
        "下载物理流场 NPZ",
        st.session_state[export_key],
        file_name="surrogate_physical_flow.npz",
        mime="application/octet-stream",
        width="stretch",
    )
    downloads[1].download_button(
        "下载叶型 JSON",
        json.dumps(prediction.blade_parameters, ensure_ascii=False, indent=2),
        file_name="candidate_blade_params.json",
        mime="application/json",
        width="stretch",
    )
    downloads[2].download_button(
        "下载指标 JSON",
        json.dumps(metrics.to_record(), ensure_ascii=False, indent=2, default=str),
        file_name="hydraulic_metrics.json",
        mime="application/json",
        width="stretch",
    )


def render_optimization(result: OptimizationResult, checkpoint_path: str, case_index: int, d_m: float, device: str) -> None:
    if result.feasible_count:
        st.success(result.message)
    else:
        st.warning(result.message)
    rows = pd.DataFrame(candidate_rows(result))
    st.dataframe(
        rows.sort_values(["feasible", "pareto", "head_m"], ascending=[False, False, False]),
        width="stretch",
        hide_index=True,
        column_config={
            "efficiency": st.column_config.NumberColumn("效率", format="%.3f"),
            "head_m": st.column_config.NumberColumn("扬程 [m]", format="%.4f"),
            "feature_velocity_m_s": st.column_config.NumberColumn("特征流速 [m/s]", format="%.4f"),
            "mask_iou": st.column_config.NumberColumn("mask IoU", format="%.5f"),
        },
    )

    valid = [(index, candidate) for index, candidate in enumerate(result.candidates) if candidate.metrics is not None]
    if valid:
        x = np.asarray([candidate.metrics.feature_velocity_m_s for _, candidate in valid])
        y = np.asarray([candidate.metrics.hydraulic_efficiency * 100.0 for _, candidate in valid])
        heads = np.asarray([candidate.metrics.total_head_m for _, candidate in valid])
        fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
        points = ax.scatter(x, y, c=heads, cmap="viridis", s=48, alpha=0.82)
        if result.pareto_indices:
            pareto = [result.candidates[index] for index in result.pareto_indices]
            ax.scatter(
                [candidate.metrics.feature_velocity_m_s for candidate in pareto],
                [candidate.metrics.hydraulic_efficiency * 100.0 for candidate in pareto],
                marker="*",
                s=170,
                facecolors="none",
                edgecolors="#ff4b4b",
                linewidths=1.5,
                label="Pareto",
            )
        if result.recommended is not None and result.recommended.metrics is not None:
            ax.scatter(
                [result.recommended.metrics.feature_velocity_m_s],
                [result.recommended.metrics.hydraulic_efficiency * 100.0],
                marker="X",
                s=120,
                c="black",
                label="推荐/最小违约",
            )
        ax.set_xlabel("混合特征流速 [m/s]（越低越好）")
        ax.set_ylabel("水力效率 [%]（越高越好）")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.colorbar(points, ax=ax, label="总扬程 [m]")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    recommended = result.recommended
    if recommended is None or recommended.metrics is None:
        return
    title = "Pareto 折中推荐" if recommended.feasible else "当前最高可行性候选（未满足扬程）"
    st.markdown(f"#### {title}")
    cols = st.columns(4)
    cols[0].metric("扬程", f"{recommended.metrics.total_head_m:.4f} m")
    cols[1].metric("效率", f"{recommended.metrics.hydraulic_efficiency:.2%}")
    cols[2].metric("特征流速", f"{recommended.metrics.feature_velocity_m_s:.4f} m/s")
    cols[3].metric("mask IoU", f"{recommended.metrics.geometry_mask_iou:.5f}")
    st.dataframe(pd.DataFrame(blade_parameter_rows(recommended.blade_parameters)), hide_index=True, width="stretch")

    actions = st.columns(2)
    actions[0].download_button(
        "下载推荐叶型 JSON",
        json.dumps(recommended.blade_parameters, ensure_ascii=False, indent=2),
        file_name="optimized_blade_params.json",
        mime="application/json",
        width="stretch",
    )
    if actions[1].button("对该候选执行完整前向复核", type="primary", width="stretch"):
        with st.spinner("正在复核完整物理流场…"):
            engine = cached_engine(checkpoint_path, case_index, device)
            prediction = engine.predict(recommended.blade_parameters, near_wall_distance_m=d_m)
        st.session_state["deployment_prediction"] = prediction
        st.session_state["deployment_prediction_label"] = title
        st.session_state["deployment_prediction_signature"] = "optimization-recheck"
        st.rerun()


st.markdown('<div class="deploy-kicker">Neural operator deployment</div>', unsafe_allow_html=True)
st.title("叶片流场部署与多目标优化")
st.caption("锁定 checkpoint 工况，仅在五层叶型内部控制点和厚度分布内部节点上做局部扰动；所有结果均为代理模型预测。")

with st.spinner("正在检查本地部署模型…"):
    catalog, rejected_models = preflight_catalog()
if not catalog:
    st.error("没有找到可读取且包含部署工况的 checkpoint。")
    if rejected_models:
        st.dataframe(pd.DataFrame(rejected_models), width="stretch", hide_index=True)
    st.stop()

st.markdown("## 1. 模型与锁定工况")
model_labels = [item["label"] for item in catalog]
selected_label = st.selectbox("部署模型", model_labels, help="列表已过滤损坏和不受支持的 checkpoint。")
selected = catalog[model_labels.index(selected_label)]
checkpoint_path = selected["path"]
metadata = cached_metadata(checkpoint_path)
case_summaries = metadata["train_case_summaries"]
case_labels = [
    f"Case {index + 1} · Q={float(case.get('qv', 0.0)):.4g} m³/s · z₀={float(case.get('z0', 0.0)):.4g} m"
    for index, case in enumerate(case_summaries)
]
case_label = st.selectbox("基准工况（选择后全部锁定）", case_labels)
case_index = case_labels.index(case_label)
metadata, case, blade_path, baseline_parameters = cached_context(checkpoint_path, case_index)

model_cols = st.columns(6)
config = metadata["model_config"]
trainer_config = metadata["trainer_config"]
model_cols[0].metric("Operator", str(config.get("operator_variant", "?")))
model_cols[1].metric("输入", str(metadata.get("input_mode", "?")))
model_cols[2].metric("Modes", str(config.get("modes", "?")))
model_cols[3].metric("Width × Depth", f"{config.get('width', '?')} × {config.get('depth', '?')}")
model_cols[4].metric("训练记录", str(metadata.get("history_length", 0)))
model_cols[5].metric("兼容状态", str(metadata.get("compatibility", "?")))
st.caption(f"基准叶型：{blade_path} · checkpoint：{checkpoint_path}")
if len({str(item.get("blade_params")) for item in case_summaries}) <= 2:
    st.warning("该 checkpoint 仅覆盖极少数叶型；结构预测属于局部 OOD 外推，推荐方案必须再做 CFD 验证。")
with st.expander("查看 checkpoint 元数据与被过滤模型"):
    st.json(
        {
            "model_config": config,
            "trainer_config": trainer_config,
            "pressure_reference_mode": config.get("pressure_reference_mode"),
            "pressure_data_reference": config.get("pressure_data_reference"),
            "train_case_count": len(case_summaries),
            "rejected_models": rejected_models,
        }
    )
render_locked_condition(case)

st.markdown("## 2. 五层结构参数")
settings = st.columns([1, 1, 2])
variation_percent = settings[0].slider("局部变化上限 ±%", 1.0, 20.0, 20.0, 0.5)
d_mm = settings[1].number_input("近壁距离 d [mm]", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
settings[2].info("可编辑：每层 2 个型线内部控制点、3 个厚度内部节点、h_max 与 t_max。默认允许相对基准 ±20%；上下表面不得相交或越出物理流道。")
variation_fraction = variation_percent / 100.0
d_m = d_mm / 1000.0

editor_key = f"blade_editor_{hashlib.sha1((checkpoint_path + str(case_index)).encode()).hexdigest()[:12]}"
reset_col, editor_note = st.columns([1, 4])
if reset_col.button("恢复基准参数", width="stretch"):
    st.session_state.pop(editor_key, None)
    st.rerun()
editor_note.caption("曲线在 BladeImport 中按最大值归一化；优化器搜索 15 个独立曲线比例自由度及 10 个 h_max/t_max 幅值，共 25 个变量。")
edited_table = st.data_editor(
    pd.DataFrame(blade_parameter_rows(baseline_parameters)),
    key=editor_key,
    hide_index=True,
    width="stretch",
    disabled=["layer"],
    num_rows="fixed",
    column_config={
        "layer": st.column_config.NumberColumn("层", disabled=True),
        "camber_1": st.column_config.NumberColumn("型线 C1", format="%.6f"),
        "camber_2": st.column_config.NumberColumn("型线 C2", format="%.6f"),
        "thickness_1": st.column_config.NumberColumn("厚度 T1", format="%.6f"),
        "thickness_2": st.column_config.NumberColumn("厚度 T2", format="%.6f"),
        "thickness_3": st.column_config.NumberColumn("厚度 T3", format="%.6f"),
        "h_max": st.column_config.NumberColumn("弯度幅值 h_max [m]", format="%.6f"),
        "t_max": st.column_config.NumberColumn("半厚度 t_max [m]", format="%.6f"),
    },
)

parameter_error: str | None = None
try:
    current_parameters = apply_blade_parameter_rows(
        baseline_parameters,
        edited_table.to_dict("records"),
        max_variation_fraction=variation_fraction,
    )
    change = shape_change_summary(current_parameters, baseline_parameters)
    st.caption(f"当前最大相对变化 {change['max_absolute_fraction']:.2%}，RMS 变化 {change['rms_fraction']:.2%}。")
except Exception as exc:
    current_parameters = baseline_parameters
    parameter_error = str(exc)
    st.error(parameter_error)

device = "cuda" if st.toggle("使用 CUDA", value=True, help="没有可用 CUDA 时后端会自动回退 CPU。") else "cpu"
predict_col, clear_col = st.columns([1, 1])
if predict_col.button("执行泛化前向预测", type="primary", width="stretch", disabled=parameter_error is not None):
    try:
        with st.spinner("正在加载模型、重建几何并映射物理流场…"):
            engine = cached_engine(checkpoint_path, case_index, device)
            prediction = engine.predict(current_parameters, near_wall_distance_m=d_m)
        st.session_state["deployment_prediction"] = prediction
        st.session_state["deployment_prediction_label"] = "当前结构参数的前向结果"
        st.session_state["deployment_prediction_signature"] = design_signature(checkpoint_path, case_index, current_parameters, d_m)
    except Exception as exc:
        st.error(f"前向预测失败：{exc}")
if clear_col.button("清除当前结果", width="stretch"):
    st.session_state.pop("deployment_prediction", None)
    st.session_state.pop("deployment_prediction_signature", None)
    st.rerun()

prediction = st.session_state.get("deployment_prediction")
if isinstance(prediction, DeploymentPrediction):
    current_signature = design_signature(checkpoint_path, case_index, current_parameters, d_m)
    stored_signature = st.session_state.get("deployment_prediction_signature")
    if stored_signature not in {current_signature, "optimization-recheck"}:
        st.info("参数或 d 已改变；下方仍显示上一次结果，请重新执行前向预测。")
    render_performance(prediction, st.session_state.get("deployment_prediction_label", "前向结果"))

st.markdown("## 3. 受约束多目标优化")
st.caption("约束 H ≥ 目标扬程；目标为最大化水力效率、最小化近壁特征流速。表面相交或越出流道始终禁止。新模型优先采用训练目录 blade_params.json 的逐参数 min/max 包络。")
training_folders = checkpoint_training_folders(checkpoint_path)
envelope_variables = []
envelope_error = None
if training_folders:
    try:
        envelope_variables = build_training_envelope_variables(baseline_parameters, training_folders)
    except Exception as exc:
        envelope_error = str(exc)
search_options = (["训练 JSON min/max 包络"] if envelope_variables else []) + ["相对基准 ±百分比"]
search_mode = st.radio("结构参数搜索域", search_options, horizontal=True)
optimization_variables = envelope_variables if search_mode == "训练 JSON min/max 包络" else None
if optimization_variables:
    unique_training_folders = {str(path.resolve()).lower() for path in training_folders}
    st.success(
        f"读取 {len(training_folders)} 条训练目录记录（{len(unique_training_folders)} 个唯一目录），"
        f"得到 {len(optimization_variables)} 个非零宽度优化变量；其余参数锁定在训练值。"
    )
    with st.expander("查看训练 JSON 参数范围"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "变量": variable.label,
                        "下限": variable.lower,
                        "基准": variable.baseline,
                        "上限": variable.upper,
                    }
                    for variable in optimization_variables
                ]
            ),
            hide_index=True,
            width="stretch",
        )
elif envelope_error:
    st.warning(f"无法建立训练 JSON 包络，已回退到百分比范围：{envelope_error}")
opt_cols = st.columns(6)
target_head = opt_cols[0].number_input("目标扬程 [m]", min_value=0.01, max_value=100.0, value=1.8, step=0.1)
population_size = int(opt_cols[1].number_input("种群", min_value=4, max_value=32, value=8, step=2))
generations = int(opt_cols[2].number_input("代数", min_value=1, max_value=10, value=3, step=1))
feature_label = opt_cols[3].selectbox("速度目标", ["混合", "平均", "P95", "最大", "RMS"])
efficiency_weight = opt_cols[4].slider("推荐中效率权重", 0.0, 1.0, 0.5, 0.05)
inference_batch = int(opt_cols[5].number_input("推理批量", min_value=1, max_value=4, value=2, step=1))
feature_metric = {"混合": "mixed", "平均": "mean", "P95": "p95", "最大": "max", "RMS": "rms"}[feature_label]
total_planned = population_size * generations
st.info(
    f"本次计划 {total_planned} 次代理推理，有效扬程约束为 H ≥ {target_head:.3f} m；"
    "若约束不可行，扫描会明确返回最高扬程候选而不会伪造 Pareto 可行解。"
)

if st.button("开始 NSGA-II 搜索", type="primary", width="stretch", disabled=parameter_error is not None):
    progress = st.progress(0.0, text="准备优化…")

    def update_progress(completed: int, total: int, stage: str) -> None:
        progress.progress(min(completed / max(total, 1), 1.0), text=f"{stage} · {completed}/{total}")

    try:
        engine = cached_engine(checkpoint_path, case_index, device)
        result = optimize_blade_design(
            EffectivePumpMetricsEngine(engine),
            baseline_parameters,
            target_head_m=float(target_head),
            variation_fraction=variation_fraction,
            near_wall_distance_m=d_m,
            population_size=population_size,
            generations=generations,
            feature_metric=feature_metric,
            efficiency_weight=efficiency_weight,
            inference_batch_size=inference_batch,
            progress_callback=update_progress,
            optimization_variables=optimization_variables,
        )
        st.session_state["deployment_optimization"] = result
        st.session_state["deployment_optimization_context"] = (checkpoint_path, case_index, d_m, device)
        progress.progress(1.0, text="优化完成")
    except Exception as exc:
        progress.empty()
        st.error(f"优化失败：{exc}")

optimization = st.session_state.get("deployment_optimization")
optimization_context = st.session_state.get("deployment_optimization_context")
if isinstance(optimization, OptimizationResult):
    if optimization_context != (checkpoint_path, case_index, d_m, device):
        st.info("下方是另一模型/工况或 d 的上一次优化结果。")
    render_optimization(optimization, checkpoint_path, case_index, d_m, device)

st.divider()
st.caption("转矩为稳态控制体角动量通量估算；效率或守恒异常时仅用于筛选。最终候选必须回到高保真 CFD/试验验证。")
