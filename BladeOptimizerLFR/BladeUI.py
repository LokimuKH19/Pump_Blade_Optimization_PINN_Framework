# blade_ui_realtime.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from BladeGeneratorCAD import BladeVoid, bezier_curve, spline_thickness, AnnularSectorPassage, generate_blade_and_fluid_domain
import pyvista as pv
from tempfile import NamedTemporaryFile
import streamlit.components.v1 as components
import base64
import os
from datetime import datetime

pv.global_theme.smooth_shading = False
pv.global_theme.multi_samples = 0

THEME_CFG = {
    "dark": {
        "fig_bg": "#0E1117",
        "ax_bg": "#0E1117",
        "fg": "#E0E0E0",
        "grid": "#444444",
        "camber": "#4DA3FF",    # 工业风格
        "thickness": "#FFB000"
    },
    "light": {
        "fig_bg": "white",
        "ax_bg": "white",
        "fg": "black",
        "grid": "#CCCCCC",
        "camber": "#1F77B4",    # 报告风格
        "thickness": "#D55E00"
    }
}


def styled_axes(theme, figsize=(6, 3)):
    cfg = THEME_CFG[theme]
    fig, ax = plt.subplots(figsize=figsize)

    fig.patch.set_facecolor(cfg["fig_bg"])
    ax.set_facecolor(cfg["ax_bg"])

    ax.tick_params(colors=cfg["fg"])
    ax.xaxis.label.set_color(cfg["fg"])
    ax.yaxis.label.set_color(cfg["fg"])
    ax.title.set_color(cfg["fg"])

    for spine in ax.spines.values():
        spine.set_color(cfg["fg"])

    ax.grid(True, linestyle="--", linewidth=0.6,
            color=cfg["grid"], alpha=0.3)

    return fig, ax


def get_theme_mode():
    return st.session_state.theme_mode


def set_mpl_theme(theme):
    if theme == "dark":
        plt.rcParams.update({
            "figure.facecolor": "#0E1117",
            "axes.facecolor": "#0E1117",
            "axes.edgecolor": "#E0E0E0",
            "axes.labelcolor": "#E0E0E0",
            "text.color": "#E0E0E0",
            "xtick.color": "#E0E0E0",
            "ytick.color": "#E0E0E0",
            "grid.color": "#444444",
            "grid.alpha": 0.3,
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "#cccccc",
            "grid.alpha": 0.3,
        })


st.set_page_config(page_title="Parametric Blade Generator", layout="wide")
st.title("🌀 Parametric Blade Generator (Realtime, Bezier + Spline)")
st.markdown("Interactive blade design with **layer-wise control** and **real-time 3D preview**.")

# ------------------ Global Blade Params ------------------

st.sidebar.header("Global Parameters")
with st.sidebar:
    theme = st.radio(
        "UI Theme",
        ["dark", "light"],
        index=0 if st.session_state.get("theme_mode", "dark") == "dark" else 1,
        key="theme_mode"
    )


def number_input_with_label(label, min_val, max_val, default_val, step=None, key=None):
    """Replace slider with number input"""
    col1, col2 = st.sidebar.columns([2, 1])
    col1.markdown(f"**{label}**")
    val = col2.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=step if step else (max_val - min_val) / 100,
        key=key,
        label_visibility="collapsed"
    )
    return val


# Global parameters in compact form
st.sidebar.subheader("Blade Geometry")
col1, col2 = st.sidebar.columns(2)
with col1:
    Theta = number_input_with_label("Theta (rad)", 0.0, np.pi, np.pi / 3.5, 0.01, "Theta")
    H = number_input_with_label("Height H (m)", 0.05, 0.5, 0.21/2, 0.01, "H")
    z0 = number_input_with_label("Z Offset (m)", -0.5, 0.5, 0.0, 0.01, "z0")
with col2:
    hub_radius = number_input_with_label("Hub Radius (m)", 0.01, 0.5, 0.121/2, 0.001, "hub")
    shroud_radius = number_input_with_label("Shroud Radius (m)", 0.01, 0.6, 0.16/2, 0.001, "shroud")
    gap_radius = number_input_with_label("Gap Radius (m)", 0.0, 10., 0., 0.001, "gap")
    points_per_chord = number_input_with_label("Chord Res", 40, 600, 80, 10, "pts")

st.sidebar.subheader("Passage Geometry")
col1, col2 = st.sidebar.columns(2)
with col1:
    N_blades = number_input_with_label(
        "Blade Count N",
        min_val=1,
        max_val=200,
        default_val=6,
        step=1,
        key="N_blades"
    )

with col2:
    H1 = number_input_with_label(
        "Passage Height H1 (m)",
        min_val=0.01,
        max_val=1.0,
        default_val=(0.21 + 0.04) / 2,   # 通常和叶高一致，工程上合理
        step=0.01,
        key="H1"
    )

# ------------------ Layer Storage ------------------
if "layers" not in st.session_state:
    st.session_state.layers = [
        {
            "mode": "hybrid",
            "theta0": 0.00,
            "h_max": 0.015,
            "t_max": 0.008,
            "camber_ctrl": [0.0, 0.45, 0.60, 0.0],
            "thickness_knots": {
                "x": [0.0, 0.05, 0.35, 0.65, 1.0],
                "t": [0.0, 0.7, 0.7, 0.3, 0.0]
            }
        },
        {
            "mode": "hybrid",
            "theta0": 0.03,
            "h_max": 0.016,
            "t_max": 0.0085,
            "camber_ctrl": [0.0, 0.48, 0.63, 0.0],
            "thickness_knots": {
                "x": [0.0, 0.05, 0.36, 0.66, 1.0],
                "t": [0.0, 0.7, 0.7, 0.3, 0.0]
            }
        },
        {
            "mode": "hybrid",
            "theta0": 0.06,
            "h_max": 0.017,
            "t_max": 0.009,
            "camber_ctrl": [0.0, 0.50, 0.65, 0.0],
            "thickness_knots": {
                "x": [0.0, 0.06, 0.37, 0.67, 1.0],
                "t": [0.0, 0.7, 0.7, 0.3, 0.0]
            }
        },
        {
            "mode": "hybrid",
            "theta0": 0.09,
            "h_max": 0.016,
            "t_max": 0.0085,
            "camber_ctrl": [0.0, 0.48, 0.63, 0.0],
            "thickness_knots": {
                "x": [0.0, 0.05, 0.36, 0.66, 1.0],
                "t": [0.0, 0.7, 0.7, 0.3, 0.0]
            }
        },
        {
            "mode": "hybrid",
            "theta0": 0.12,
            "h_max": 0.015,
            "t_max": 0.008,
            "camber_ctrl": [0.0, 0.45, 0.60, 0.0],
            "thickness_knots": {
                "x": [0.0, 0.05, 0.35, 0.65, 1.0],
                "t": [0.0, 0.7, 0.7, 0.3, 0.0]
            }
        }
    ]

layers = st.session_state.layers
xi = np.linspace(0, 1, points_per_chord)

# ------------------ Layer Editor ------------------
layer_tabs = st.tabs([f"Layer {i + 1}" for i in range(5)])

for i, tab in enumerate(layer_tabs):
    with tab:
        prm = layers[i]

        # Create two main columns
        col1, col2 = st.columns(2)

        # --- Left Column: Camber Control ---
        with col1:
            st.subheader("Camber γ(x) — Bezier Control")

            # Compact parameter input for camber
            st.markdown("**Control Points**")
            cam_cols = st.columns(4)
            with cam_cols[0]:
                st.markdown("**P₀** = 0.0")
            with cam_cols[1]:
                c1 = st.number_input(
                    f"P₁ (Layer {i + 1})",
                    min_value=0.0,
                    max_value=1.2,
                    value=float(prm["camber_ctrl"][1]),
                    step=0.01,
                    key=f"c1_{i}",
                    format="%.3f"
                )
            with cam_cols[2]:
                c2 = st.number_input(
                    f"P₂ (Layer {i + 1})",
                    min_value=0.0,
                    max_value=1.2,
                    value=float(prm["camber_ctrl"][2]),
                    step=0.01,
                    key=f"c2_{i}",
                    format="%.3f"
                )
            with cam_cols[3]:
                st.markdown("**P₃** = 0.0")

            prm["camber_ctrl"] = [0.0, c1, c2, 0.0]
            st.markdown(r"**$\gamma$ and $\tau$ are normalized**")

            # Camber plot
            gamma = bezier_curve(xi, prm["camber_ctrl"])
            theme = st.session_state.theme_mode
            cfg = THEME_CFG[theme]

            fig, ax = styled_axes(theme)
            ax.plot(xi, gamma, lw=2.2, color=cfg["camber"])
            ax.set_xlabel("x")
            ax.set_ylabel("γ(x)")
            ax.set_title(f"Layer {i + 1} Camber Profile")

            st.pyplot(fig, width="stretch")

        # --- Right Column: Thickness Control ---
        with col2:
            st.subheader("Thickness τ(x) — Spline Control")

            # Compact parameter input for thickness
            st.markdown("**Thickness Knot Values**")
            thick_cols = st.columns(5)
            thickness_knots = prm["thickness_knots"]

            # Fixed positions (can make these editable if needed)
            with thick_cols[0]:
                st.markdown("**x₀** = 0.0")
                st.markdown("**τ₀** = 0.0")

            with thick_cols[1]:
                st.markdown(f"**x₁** = {thickness_knots['x'][1]:.2f}")
                t1 = st.number_input(
                    f"τ₁ (L{i + 1})",
                    min_value=0.1,
                    max_value=1.5,
                    value=float(thickness_knots["t"][1]),
                    step=0.01,
                    key=f"t1_{i}",
                    format="%.3f"
                )

            with thick_cols[2]:
                st.markdown(f"**x₂** = {thickness_knots['x'][2]:.2f}")
                t2 = st.number_input(
                    f"τ₂ (L{i + 1})",
                    min_value=0.1,
                    max_value=1.5,
                    value=float(thickness_knots["t"][2]),
                    step=0.01,
                    key=f"t2_{i}",
                    format="%.3f"
                )

            with thick_cols[3]:
                st.markdown(f"**x₃** = {thickness_knots['x'][3]:.2f}")
                t3 = st.number_input(
                    f"τ₃ (L{i + 1})",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(thickness_knots["t"][3]),
                    step=0.01,
                    key=f"t3_{i}",
                    format="%.3f"
                )

            with thick_cols[4]:
                st.markdown("**x₄** = 1.0")
                st.markdown("**τ₄** = 0.0")

            prm["thickness_knots"]["t"] = [0.0, t1, t2, t3, 0.0]
            # Thickness plot
            tau = spline_thickness(xi, prm["thickness_knots"]["x"], prm["thickness_knots"]["t"])
            theme = st.session_state.theme_mode
            cfg = THEME_CFG[theme]

            fig, ax = styled_axes(theme)
            ax.plot(xi, tau, lw=2.2, color=cfg["thickness"])
            ax.set_xlabel("x")
            ax.set_ylabel("τ(x)")
            ax.set_title(f"Layer {i + 1} Thickness Profile")

            st.pyplot(fig, width="stretch")

        # --- Scalar Parameters (Compact Row) ---
        st.markdown("---")
        st.subheader("Scalar Parameters")

        # 创建参数表格，每行一个参数
        scalar_params = [
            {
                "name": "θ₀",
                "description": "Base angle",
                "unit": "rad",
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
                "value": prm["theta0"],
                "key": f"theta0_{i}",
                "format": "%.3f"
            },
            {
                "name": "h_max",
                "description": "Max camber height",
                "unit": "m",
                "min": 0.0,
                "max": 0.05,
                "step": 0.001,
                "value": prm["h_max"],
                "key": f"hmax_{i}",
                "format": "%.4f"
            },
            {
                "name": "t_max",
                "description": "Max thickness",
                "unit": "m",
                "min": 0.0,
                "max": 0.05,
                "step": 0.001,
                "value": prm["t_max"],
                "key": f"tmax_{i}",
                "format": "%.4f"
            }
        ]

        # 表头
        header_cols = st.columns([1, 2, 2, 1, 1, 1])
        with header_cols[0]:
            st.markdown("**Parameter**")
        with header_cols[1]:
            st.markdown("**Description**")
        with header_cols[2]:
            st.markdown("**Value**")
        with header_cols[3]:
            st.markdown("**Unit**")
        with header_cols[4]:
            st.markdown("**Min**")
        with header_cols[5]:
            st.markdown("**Max**")

        for param in scalar_params:
            row_cols = st.columns([1, 2, 2, 1, 1, 1])

            with row_cols[0]:
                st.markdown(f"**{param['name']} (L{i+1})**")

            with row_cols[1]:
                st.markdown(param['description'])

            with row_cols[2]:
                col1, col2 = st.columns([4, 1])
                with col1:
                    val = st.number_input(
                        param['description'],
                        min_value=param['min'],
                        max_value=param['max'],
                        value=float(param['value']),
                        step=param['step'],
                        key=param['key'],
                        format=param['format'],
                        label_visibility="collapsed"
                    )
                with col2:
                    pass

                # 更新参数值
                if param['name'] == "θ₀":
                    prm["theta0"] = val
                elif param['name'] == "h_max":
                    prm["h_max"] = val
                elif param['name'] == "t_max":
                    prm["t_max"] = val

            with row_cols[3]:
                st.markdown(param['unit'])

            with row_cols[4]:
                st.markdown(f"{param['min']}")

            with row_cols[5]:
                st.markdown(f"{param['max']}")

            st.markdown("<hr style='margin: 0.2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

# ------------------ Blade Preview ------------------
st.markdown("---")
st.header("🔍 3D Blade Preview (Interactive)")
st.markdown("### Interactive 3D Viewer")


# ================== Theme-adaptive colors ==================
if theme == "dark":
    pv_bg_color = "#0E1117"      # Streamlit dark background
    blade_color = "#FFB000"      # bright orange
    edge_color = "#333333"
    text_color = "white"
else:
    pv_bg_color = "white"
    blade_color = "#D55E00"      # deep orange
    edge_color = "#666666"
    text_color = "black"


# Generate Passage
passage = AnnularSectorPassage(
    hub_radius=hub_radius,
    shroud_radius=shroud_radius,
    z0=z0,
    H1=H1,
    N=N_blades,
    nr=40,
    ntheta=40,
    nz=60
)
passage_grid, z_lim, theta_lim = passage.generate_surface()

# Generate blade
blade = BladeVoid(
    span_layers=layers,
    Theta=Theta,
    H=H,
    z0=z0,
    hub_radius=hub_radius,
    shroud_radius=shroud_radius,
    theta_offset=0.0
)
blade.generate_surface(passage_z=z_lim, passage_theta=theta_lim, align="left")

plotter = blade.visualize_streamlit(
    passage_grid=passage_grid,
    theme=theme
)


# ✅ 正确的 PyVista → Streamlit 嵌入方式
html = plotter.export_html(None)

with NamedTemporaryFile(delete=False, suffix=".html") as f:
    plotter.export_html(f.name)
    f.seek(0)
    html = f.read().decode("utf-8")

components.html(html, height=650, scrolling=False)

# ------------------ Export ------------------
st.markdown("---")
st.header("💾 Export Settings")

# ------------------ Current Parameters Summary ------------------
with st.expander("📋 Current Parameters Summary"):
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.subheader("Global Parameters")
        st.write(f"- Theta: {Theta:.3f} rad")
        st.write(f"- Height H: {H:.3f} m")
        st.write(f"- Z Offset: {z0:.3f} m")
        st.write(f"- Hub Radius: {hub_radius:.3f} m")
        st.write(f"- Shroud Radius: {shroud_radius:.3f} m")
        st.write(f"- Gap Radius: {gap_radius:.3f} m")
        st.write(f"- Chord Resolution: {points_per_chord}")

    with summary_cols[1]:
        st.subheader("Layer Overview")
        for i, layer in enumerate(layers):
            st.write(
                f"**Layer {i + 1}**: θ₀={layer['theta0']:.3f}, h_max={layer['h_max']:.4f}, t_max={layer['t_max']:.4f}")

export_cols = st.columns(2)

with export_cols[0]:
    export_dir = st.text_input(
        "Export Dir",
        value="CQ",
        help="Directory name (ASCII only)"
    )

with export_cols[1]:
    add_timestamp = st.checkbox(
        "Append timestamp",
        value=True,
        help="Append time suffix like _20260108_143210"
    )


if st.button("⬇️ Export Blade Model", type="primary"):
    try:
        # --------- 生成输出目录名 ---------
        if add_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./{export_dir}_{ts}"
        else:
            output_dir = f"./{export_dir}"

        os.makedirs(output_dir, exist_ok=True)

        # --------- 组织参数数据 ---------
        param_data = {
            "layers_params": layers
        }

        # --------- 调用主生成函数 ---------
        generate_blade_and_fluid_domain(
            param_data=param_data,
            hub_radius=hub_radius,
            shroud_radius=shroud_radius,
            N=N_blades,
            H1=H1,
            H=H,
            z0=z0,
            Theta=Theta,
            output_dir=output_dir,
            preview=False,   # UI 下不建议开 OCC 预览
            gap=gap_radius,
        )

        st.success(f"✅ Export completed\n\nOutput directory:\n`{output_dir}`")

    except Exception as e:
        st.error(f"Export failed: {str(e)}")
