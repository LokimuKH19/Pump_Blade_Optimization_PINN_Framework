# blade_ui_realtime.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from BladeGenerator import Blade3D, bezier_curve, spline_thickness
import pyvista as pv
from tempfile import NamedTemporaryFile
import base64



st.set_page_config(page_title="Parametric Blade Generator", layout="wide")
st.title("🌀 Parametric Blade Generator (Realtime, Bezier + Spline)")
st.markdown("Interactive blade design with **layer-wise control** and **real-time 3D preview**.")

# ------------------ Global Blade Params ------------------
st.sidebar.header("Global Parameters")


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
    Theta = number_input_with_label("Theta (rad)", 0.0, np.pi, 0.52, 0.01, "Theta")
    H = number_input_with_label("Height H (m)", 0.05, 0.5, 0.21, 0.01, "H")
    z0 = number_input_with_label("Z Offset (m)", -0.5, 0.5, -0.10, 0.01, "z0")
with col2:
    hub_radius = number_input_with_label("Hub Radius (m)", 0.01, 0.5, 0.121, 0.001, "hub")
    shroud_radius = number_input_with_label("Shroud Radius (m)", 0.01, 0.6, 0.16, 0.001, "shroud")
    points_per_chord = number_input_with_label("Chord Res", 100, 600, 300, 10, "pts")

# ------------------ Layer Storage ------------------
if "layers" not in st.session_state:
    st.session_state.layers = [
        {
            "mode": "hybrid",
            "theta0": 0.00,
            "h_max": 0.022,
            "t_max": 0.013,
            "camber_ctrl": [0.0, 0.60, 0.85, 0.0],
            "thickness_knots": {"x": [0.0, 0.05, 0.30, 0.75, 1.0], "t": [0.0, 1.0, 1.0, 0.40, 0.0]}
        },
        {
            "mode": "hybrid",
            "theta0": 0.04,
            "h_max": 0.024,
            "t_max": 0.014,
            "camber_ctrl": [0.0, 0.62, 0.88, 0.0],
            "thickness_knots": {"x": [0.0, 0.05, 0.32, 0.75, 1.0], "t": [0.0, 1.0, 1.0, 0.38, 0.0]}
        },
        {
            "mode": "hybrid",
            "theta0": 0.08,
            "h_max": 0.026,
            "t_max": 0.015,
            "camber_ctrl": [0.0, 0.65, 0.92, 0.0],
            "thickness_knots": {"x": [0.0, 0.06, 0.35, 0.78, 1.0], "t": [0.0, 1.0, 1.0, 0.35, 0.0]}
        },
        {
            "mode": "hybrid",
            "theta0": 0.12,
            "h_max": 0.022,
            "t_max": 0.013,
            "camber_ctrl": [0.0, 0.60, 0.88, 0.0],
            "thickness_knots": {"x": [0.0, 0.06, 0.32, 0.75, 1.0], "t": [0.0, 1.0, 1.0, 0.32, 0.0]}
        },
        {
            "mode": "hybrid",
            "theta0": 0.16,
            "h_max": 0.018,
            "t_max": 0.011,
            "camber_ctrl": [0.0, 0.55, 0.80, 0.0],
            "thickness_knots": {"x": [0.0, 0.05, 0.28, 0.70, 1.0], "t": [0.0, 1.0, 1.0, 0.30, 0.0]}
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
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(xi, gamma, lw=2)
            ax.set_xlabel("x")
            ax.set_ylabel("γ(x)")
            ax.set_title(f"Layer {i + 1} Camber Profile")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

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
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(xi, tau, lw=2, color="orange")
            ax.set_xlabel("x")
            ax.set_ylabel("τ(x)")
            ax.set_title(f"Layer {i + 1} Thickness Profile")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)

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

# Generate blade
blade = Blade3D(
    span_layers=layers,
    Theta=Theta,
    H=H,
    z0=z0,
    hub_radius=hub_radius,
    shroud_radius=shroud_radius,
)
blade.generate_surface(points_per_chord=points_per_chord)

# 创建PyVista plotter
plotter = pv.Plotter(window_size=[800, 600])

# 获取网格
try:
    mesh = blade.to_pyvista_mesh()

    # 添加网格到plotter
    plotter.add_mesh(
        mesh,
        color='lightblue',
        show_edges=True,
        opacity=0.9,
        smooth_shading=True
    )

    # 设置相机视角
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.5)

    # 添加坐标轴
    plotter.add_axes()

    # 添加标题
    plotter.add_title("3D Blade Preview", font_size=20)

    # 方法1: 导出为HTML并嵌入
    st.markdown("### Interactive 3D Viewer")

    # 创建临时HTML文件
    with NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
        html_file = tmp_file.name

    # 导出场景为HTML
    plotter.export_html(html_file)

    # 读取HTML内容
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # 嵌入HTML到Streamlit
    st.components.v1.html(html_content, height=600, scrolling=False)

    # 添加一些交互控制
    with st.expander("3D View Controls"):
        st.markdown("""
        **Mouse Controls:**
        - **Left-click + drag**: Rotate view
        - **Right-click + drag**: Pan view  
        - **Scroll wheel**: Zoom in/out
        - **Middle-click + drag**: Change field of view

        **Keyboard Shortcuts:**
        - **r**: Reset camera
        - **c**: Toggle axes visibility
        - **s**: Save screenshot
        """)

        # 添加一些视图预设按钮
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Front View"):
                plotter.camera_position = 'xy'
        with col2:
            if st.button("Side View"):
                plotter.camera_position = 'yz'
        with col3:
            if st.button("Top View"):
                plotter.camera_position = 'xz'
        with col4:
            if st.button("Isometric"):
                plotter.camera_position = 'iso'

except Exception as e:
    st.error(f"Failed to generate 3D preview: {str(e)}")
    st.info("Make sure PyVista is installed: `pip install pyvista`")

    # 回退方案：静态图像
    try:
        screenshot = mesh.plot(off_screen=True, screenshot=True,
                               window_size=[800, 600], show_edges=True)
        st.image(screenshot, caption="Static 3D Preview")
    except:
        st.warning("Could not generate any 3D visualization.")

# 关闭plotter
plotter.close()

# ------------------ Export ------------------
st.markdown("---")
st.header("💾 Export Settings")

export_cols = st.columns(3)
with export_cols[0]:
    filename = st.text_input("Export Filename", value="blade_ui_realtime")
with export_cols[1]:
    export_format = st.selectbox("Export Format", ["STL", "JSON", "Both"])
with export_cols[2]:
    as_solid = st.checkbox("Export as Solid", value=True)

if st.button("⬇️ Export Blade Model", type="primary"):
    try:
        blade.export_mesh(
            f"./Blades/{filename}",
            mode=export_format.lower(),
            as_solid=as_solid,
            save_params=True,
        )
        st.success(f"✅ Export completed: {filename}.stl & {filename}.json")
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

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
        st.write(f"- Chord Resolution: {points_per_chord}")

    with summary_cols[1]:
        st.subheader("Layer Overview")
        for i, layer in enumerate(layers):
            st.write(
                f"**Layer {i + 1}**: θ₀={layer['theta0']:.3f}, h_max={layer['h_max']:.4f}, t_max={layer['t_max']:.4f}")