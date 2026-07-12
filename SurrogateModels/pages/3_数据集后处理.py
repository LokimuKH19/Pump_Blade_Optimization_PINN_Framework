from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SurrogateDatasetPostprocess import (  # noqa: E402
    DatasetBatchPostprocessResult,
    postprocess_dataset_folders,
)
from SurrogateDeployment import physical_coordinates  # noqa: E402


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_FOLDERS = [
    ROOT.parent / "BladeOptimizerLFR" / "CQ_20260514_115826_SIMULATION",
    ROOT.parent / "BladeOptimizerLFR" / "CQ_20260519_160122_S01",
    ROOT.parent / "BladeOptimizerLFR" / "CQ_20260514_115826_SIMULATION",
]


def _summary_table(batch: DatasetBatchPostprocessResult) -> pd.DataFrame:
    rows = []
    for result in batch.results:
        metrics = result.metrics
        rows.append(
            {
                "记录": result.record_index + 1,
                "数据目录": result.resolved_path.name,
                "重复": "是" if result.duplicate else "否",
                "场来源": result.source_kind,
                "成功": result.success,
                "signed扬程 [m]": np.nan if metrics is None else metrics.total_head_m,
                "有效扬程 [m]": np.nan if metrics is None else -metrics.total_head_m,
                "差压 Δp [kPa]": np.nan if metrics is None else metrics.static_pressure_rise_pa / 1000.0,
                "signed效率 [%]": np.nan if metrics is None else metrics.hydraulic_efficiency * 100.0,
                "有效效率 [%]": np.nan if metrics is None else -metrics.hydraulic_efficiency * 100.0,
                "特征流速 [m/s]": np.nan if metrics is None else metrics.feature_velocity_m_s,
                "转矩 [N·m]": np.nan if metrics is None else metrics.driving_torque_n_m,
                "轴功率 [W]": np.nan if metrics is None else metrics.shaft_input_power_w,
                "水力功率 [W]": np.nan if metrics is None else metrics.hydraulic_output_power_w,
                "质量不平衡 [%]": np.nan if metrics is None else metrics.mass_imbalance_ratio * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _render_field(result) -> None:
    if result.fields_physical is None or result.blade_mask is None:
        return
    field_options = {
        "径向速度 UR [m/s]": ("UR", 1.0),
        "周向速度 UT [m/s]": ("UT", 1.0),
        "轴向速度 UZ [m/s]": ("UZ", 1.0),
        "压力 P [kPa]": ("P", 1e-3),
    }
    controls = st.columns(2)
    label = controls[0].selectbox("物理场", list(field_options), key=f"dataset_field_{result.record_index}")
    span = controls[1].slider("径向 span", 0.0, 1.0, 0.5, 0.05, key=f"dataset_span_{result.record_index}")
    name, scale = field_options[label]
    field = np.asarray(result.fields_physical[name], dtype=float)
    radial_index = int(round(span * (field.shape[0] - 1)))
    coordinates = physical_coordinates(result.case)
    radius = coordinates.r_m[radial_index]
    arc_mm = radius * coordinates.theta_rad * 1000.0
    z_mm = coordinates.z_m * 1000.0
    blade = np.asarray(result.blade_mask[radial_index].T, dtype=bool)
    values = np.ma.masked_where(blade, field[radial_index].T * scale)
    xx, zz = np.meshgrid(arc_mm, z_mm)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    ax.set_facecolor("#e6e6e6")
    image = ax.pcolormesh(xx, zz, values, shading="auto", cmap=cmap)
    ax.contour(xx, zz, blade.astype(float), levels=[0.5], colors="black", linewidths=1.0)
    ax.set_xlabel("物理周向弧长 rθ [mm]")
    ax.set_ylabel("物理轴向坐标 z [mm]")
    ax.set_title(f"{result.resolved_path.name} · {label} · span={span:.2f}")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("灰色区域为 BladeImport/IBM 固体掩膜，不含流场值。")


st.set_page_config(page_title="数据集后处理与外特性验证", page_icon="📂", layout="wide")
st.title("📂 数据文件夹观察与外特性验证")
st.write(
    "输入1–3个CFD或代理数据目录。页面会自动发现 blade_params.json 和 Fluent CSV/NPZ，"
    "映射到统一物理网格，并用与部署模型相同的公式计算扬程、差压、效率、功率、转矩和近壁特征流速。"
)

folder_text = st.text_area(
    "数据目录（每行一个，最多3条；重复路径会保留并标注）",
    value="\n".join(str(path.resolve()) for path in DEFAULT_FOLDERS),
    height=125,
)
settings = st.columns(3)
d_mm = settings[0].number_input("近壁距离 d [mm]", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
grid_size = int(settings[1].selectbox("统一物理网格", [32, 48, 64], index=2))
settings[2].info("Fluent CSV首次处理需要散点插值；重复目录会复用第一次结果。")

folders = [line.strip() for line in folder_text.splitlines() if line.strip()]
if st.button("扫描目录并计算外特性", type="primary", width="stretch", disabled=not (1 <= len(folders) <= 3)):
    with st.spinner("正在发现数据、插值物理场并执行水力后处理……"):
        batch = postprocess_dataset_folders(
            folders,
            near_wall_distance_m=float(d_mm) / 1000.0,
            grid_size=grid_size,
        )
    st.session_state["dataset_postprocess_batch"] = batch

batch = st.session_state.get("dataset_postprocess_batch")
if isinstance(batch, DatasetBatchPostprocessResult):
    st.subheader("外特性汇总")
    summary = _summary_table(batch)
    st.dataframe(summary, hide_index=True, width="stretch")
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "下载外特性 CSV",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="dataset_external_characteristics.csv",
        mime="text/csv",
        width="stretch",
    )
    raw_summary = batch.summary_records()
    download_cols[1].download_button(
        "下载完整诊断 JSON",
        json.dumps(raw_summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name="dataset_postprocess_summary.json",
        mime="application/json",
        width="stretch",
    )

    st.subheader("数据集对比")
    successful = summary[summary["成功"]].copy()
    if not successful.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
        x = np.arange(len(successful))
        names = [f"记录 {int(value)}" for value in successful["记录"]]
        for ax, column, title in zip(
            axes,
            ["有效扬程 [m]", "有效效率 [%]", "特征流速 [m/s]"],
            ["总扬程", "水力效率", "近壁混合特征流速"],
        ):
            ax.bar(x, successful[column].to_numpy(float), color="#2c7fb8")
            ax.set_xticks(x, names)
            ax.set_title(title)
            ax.set_ylabel(column)
            ax.grid(axis="y", alpha=0.2)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    tabs = st.tabs([f"记录 {item.record_index + 1}" for item in batch.results])
    for tab, result in zip(tabs, batch.results):
        with tab:
            if result.duplicate:
                st.info(f"该路径与记录 {result.duplicate_of_index + 1} 重复；本条保留但复用首次计算结果。")
            if result.metrics is not None:
                metrics = result.metrics
                cols = st.columns(6)
                cols[0].metric("有效扬程", f"{-metrics.total_head_m:.5f} m", help=f"原始 signed H={metrics.total_head_m:.5f} m")
                cols[1].metric("有效水力效率", f"{-metrics.hydraulic_efficiency:.3%}", help=f"原始 signed η={metrics.hydraulic_efficiency:.3%}")
                cols[2].metric("差压", f"{metrics.static_pressure_rise_pa/1000:.3f} kPa")
                cols[3].metric("特征流速", f"{metrics.feature_velocity_m_s:.4f} m/s")
                cols[4].metric("驱动转矩", f"{metrics.driving_torque_n_m:.4f} N·m")
                cols[5].metric("水力功率", f"{metrics.hydraulic_output_power_w:.2f} W")
                _render_field(result)
            diagnostics = pd.DataFrame([item.to_record() for item in result.diagnostics])
            if not diagnostics.empty:
                st.markdown("**自动诊断**")
                st.dataframe(diagnostics, hide_index=True, width="stretch")
else:
    st.caption("尚未计算。可直接保留默认三条路径进行验证。")
