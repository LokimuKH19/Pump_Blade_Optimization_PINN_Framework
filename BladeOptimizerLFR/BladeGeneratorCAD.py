import json
import numpy as np
import math
from scipy.interpolate import PchipInterpolator, CubicSpline
import scipy.special
import pyvista as pv
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Vec, gp_Pnt, gp_Circ
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
from OCC.Core.TopTools import TopTools_ListOfShape


# ---------------- 流道 -----------------
class AnnularSectorPassage:
    def __init__(self, hub_radius, shroud_radius, z0, H1, N, nr=40, ntheta=40, nz=60):
        self.Rh = hub_radius
        self.Rs = shroud_radius
        self.z0 = z0
        self.H1 = H1
        self.N = N
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz

    def generate_surface(self):
        r = np.linspace(self.Rh, self.Rs, self.nr)
        theta = np.linspace(0, 2*np.pi/self.N, self.ntheta)
        z = np.linspace(self.z0, self.z0+self.H1, self.nz)
        R, Theta, Z = np.meshgrid(r, theta, z, indexing="ij")
        X = R*np.cos(Theta)
        Y = R*np.sin(Theta)
        return pv.StructuredGrid(X,Y,Z), (self.z0, self.z0+self.H1), (0, 2*np.pi/self.N)


def make_sector_prism(radius, height, angle):
    p0 = gp_Pnt(0, 0, 0)
    p1 = gp_Pnt(radius, 0, 0)
    p2 = gp_Pnt(radius * math.cos(angle), radius * math.sin(angle), 0)
    center = gp_Pnt(0, 0, 0)
    axis = gp_Ax2(center, gp_Dir(0, 0, 1))
    circle = gp_Circ(axis, radius)
    arc_edge = BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(circle, 0.0, angle, True).Value()).Edge()
    wire_builder = BRepBuilderAPI_MakeWire()
    wire_builder.Add(BRepBuilderAPI_MakeEdge(p0, p1).Edge())
    wire_builder.Add(arc_edge)
    wire_builder.Add(BRepBuilderAPI_MakeEdge(p2, p0).Edge())
    if not wire_builder.IsDone():
        wire_builder.Add(BRepBuilderAPI_MakeEdge(p1, p2).Edge())
    wire = wire_builder.Wire()
    face_builder = BRepBuilderAPI_MakeFace(wire)
    if not face_builder.IsDone():
        raise RuntimeError("Failed to create face from wire")
    face = face_builder.Face()
    prism = BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, height))
    if not prism.IsDone():
        raise RuntimeError("Failed to create prism")
    return prism.Shape()


# 或者更简单的版本 - 直接使用扇柱减去内圆柱
def make_annular_sector_prism_simple(r_inner, r_outer, height, angle):
    """
    简化的方法：创建一个完整的外扇柱，减去一个完整的内扇柱
    更可靠，确保是闭合实体
    """
    # 创建外扇柱
    outer_sector = make_sector_prism(r_outer, height, angle)

    # 创建内扇柱
    inner_sector = make_sector_prism(r_inner, height, angle)

    # 执行布尔差集：外扇柱 - 内扇柱
    cutter = BRepAlgoAPI_Cut(outer_sector, inner_sector)
    if not cutter.IsDone():
        raise RuntimeError("Failed to create annular sector via boolean operation")

    return cutter.Shape()


# ---------------- Bezier / Thickness -----------------
def bezier_curve(x, ctrl):
    x = np.asarray(x)
    n = len(ctrl) - 1
    ctrl = np.asarray(ctrl, dtype=float)
    gamma = np.zeros_like(x)
    for i in range(n + 1):
        gamma += scipy.special.comb(n, i) * (x ** i) * ((1 - x) ** (n - i)) * ctrl[i]
    maxv = gamma.max()
    if maxv > 1e-12:
        gamma /= maxv
    return gamma

def spline_thickness(x, knots_x, knots_t):
    cs = CubicSpline(knots_x, knots_t)
    tau = cs(x)
    tau = np.clip(tau, 0.0, None)
    maxv = tau.max()
    if maxv > 1e-12:
        tau /= maxv
    return tau

# ---------------- BladeVoid -----------------
class BladeVoid:
    def __init__(self, span_layers, Theta, H, z0, hub_radius, shroud_radius, theta_offset=0.0):
        self.layers = span_layers
        self.Theta = float(Theta)
        self.H = float(H)
        self.z0 = float(z0)
        self.hub_radius = float(hub_radius)
        self.shroud_radius = float(shroud_radius)
        self.theta_offset = float(theta_offset)

        self.vertices_upper = None
        self.vertices_lower = None

        self._build_spanwise_interpolators()
        self._build_spanwise_shape_interpolators()

    def _build_spanwise_interpolators(self):
        n = len(self.layers)
        s = np.linspace(0,1,n)
        def collect(key):
            return np.array([float(li[key]) for li in self.layers])
        self._theta0_s = PchipInterpolator(s, collect("theta0"))
        self._hmax_s = PchipInterpolator(s, collect("h_max"))
        self._tmax_s = PchipInterpolator(s, collect("t_max"))
        radius = []
        for i, li in enumerate(self.layers):
            if "radius" in li:
                radius.append(li["radius"])
            else:
                w = i/(n-1)
                radius.append((1-w)*self.hub_radius + w*self.shroud_radius)
        self._radius_s = PchipInterpolator(s, np.asarray(radius))

    def _build_spanwise_shape_interpolators(self):
        n = len(self.layers)
        s = np.linspace(0,1,n)
        camber_layers = np.array([li["camber_ctrl"] for li in self.layers])
        Nc = camber_layers.shape[1]
        self._camber_ctrl_s = [PchipInterpolator(s, camber_layers[:,k]) for k in range(Nc)]
        knots_x = self.layers[0]["thickness_knots"]["x"]
        thickness_layers = np.array([li["thickness_knots"]["t"] for li in self.layers])
        Nt = thickness_layers.shape[1]
        self._thickness_ctrl_s = [PchipInterpolator(s, thickness_layers[:,j]) for j in range(Nt)]
        self._thickness_knots_x = np.asarray(knots_x)

    # ---------------- PyVista 可视化 -----------------
    def generate_surface(self, chord_pts=50, span_pts=10, passage_z=None, passage_theta=None, align="left"):
        xi = np.linspace(0,1,chord_pts)
        s_vals = np.linspace(0,1,span_pts)
        verts_upper = np.zeros((span_pts, chord_pts, 3))
        verts_lower = np.zeros((span_pts, chord_pts, 3))

        if passage_z is not None:
            z0_passage, z1_passage = passage_z
            blade_z_center = (z0_passage + z1_passage)/2
            self.z0 = blade_z_center - self.H/2

        if passage_theta is not None:
            theta0_passage, theta1_passage = passage_theta
            if align=="center":
                blade_theta_center = (theta0_passage + theta1_passage)/2
                self.theta_offset = blade_theta_center - 0.5*self.Theta
            elif align=="left":
                self.theta_offset = theta0_passage

        for i, s in enumerate(s_vals):
            theta0 = float(self._theta0_s(s))
            h_max = float(self._hmax_s(s))
            t_max = float(self._tmax_s(s))
            R = float(self._radius_s(s))
            camber_ctrl = np.array([f(s) for f in self._camber_ctrl_s])
            gamma = bezier_curve(xi, camber_ctrl)
            t_ctrl = np.array([f(s) for f in self._thickness_ctrl_s])
            tau = spline_thickness(xi, self._thickness_knots_x, t_ctrl)

            for j, x in enumerate(xi):
                theta = self.theta_offset + theta0 + x*self.Theta
                zc = self.z0 + x*self.H - h_max*gamma[j]
                zu = zc + t_max*tau[j]
                zl = zc - t_max*tau[j]
                verts_upper[i,j] = [R*np.cos(theta), R*np.sin(theta), zu]
                verts_lower[i,j] = [R*np.cos(theta), R*np.sin(theta), zl]

        self.vertices_upper = verts_upper
        self.vertices_lower = verts_lower

    def visualize(self, passage_grid=None):
        if self.vertices_upper is None:
            self.generate_surface()
        up_pts = self.vertices_upper.reshape(-1,3)
        low_pts = self.vertices_lower.reshape(-1,3)
        span_pts, chord_pts, _ = self.vertices_upper.shape

        surf_upper = pv.StructuredGrid()
        surf_upper.points = up_pts
        surf_upper.dimensions = (chord_pts, span_pts, 1)
        surf_lower = pv.StructuredGrid()
        surf_lower.points = low_pts
        surf_lower.dimensions = (chord_pts, span_pts, 1)

        p = pv.Plotter()
        if passage_grid is not None:
            p.add_mesh(passage_grid, color="lightgray", opacity=0.3, show_edges=True)
        p.add_mesh(surf_upper, color="tomato", show_edges=True)
        p.add_mesh(surf_lower, color="tomato", show_edges=True)

        # hub/shroud 可视化
        z_root = self.vertices_upper[0,0,2]
        z_tip  = self.vertices_upper[-1,0,2]
        for r in [self.hub_radius, self.shroud_radius]:
            theta = np.linspace(0,2*np.pi,100)
            x = r*np.cos(theta)
            y = r*np.sin(theta)
            z_root_array = np.full_like(theta, z_root)
            z_tip_array  = np.full_like(theta, z_tip)
            p.add_mesh(np.column_stack([x,y,z_root_array]), color="blue", point_size=5, render_points_as_spheres=True)
            p.add_mesh(np.column_stack([x,y,z_tip_array]), color="green", point_size=5, render_points_as_spheres=True)

        p.add_axes()
        p.show()

    # ---------------- OCCT Solid -----------------
    def to_occt_solid_loft(self):
        if self.vertices_upper is None:
            self.generate_surface()

        span_pts, chord_pts, _ = self.vertices_upper.shape
        loft = BRepOffsetAPI_ThruSections(True, True, 1e-6)

        for i in range(span_pts):
            polygon_builder = BRepBuilderAPI_MakePolygon()
            for j in range(chord_pts):
                x, y, z = self.vertices_upper[i, j]
                polygon_builder.Add(gp_Pnt(x, y, z))
            for j in range(chord_pts - 1, -1, -1):
                x, y, z = self.vertices_lower[i, j]
                polygon_builder.Add(gp_Pnt(x, y, z))
            # 闭合多边形
            x_first, y_first, z_first = self.vertices_upper[i, 0]
            polygon_builder.Add(gp_Pnt(x_first, y_first, z_first))

            if polygon_builder.IsDone():
                wire = polygon_builder.Wire()
                loft.AddWire(wire)
            else:
                print(f"Warning: Failed to create polygon for span layer {i}")

        loft.Build()
        if not loft.IsDone():
            raise RuntimeError("Loft failed to build solid")
        return loft.Shape()

    # ---------------- BladeVoid 类中的新方法，等主程序调通启用 ----------------
    def _summon_hyp_loft(self, rate=1.5):
        # 创建假想叶片实体：半径从shroud_radius向外延伸，由于左侧的稳定特性和右侧的奇异特性的必要举措
        # 我们创建一个虚拟的span层，半径大于shroud_radius
        shroud_radius_plus = self.shroud_radius*rate
        # 我们只需要两个span层：一个在shroud_radius，一个在shroud_radius_plus
        # 保持弦向点数量和参数与原始叶片一致

        # 获取原始叶片的弦向参数
        chord_pts = self.vertices_upper.shape[1]
        xi = np.linspace(0, 1, chord_pts)

        # 获取shroud处的叶片参数
        s_shroud = 1.0  # shroud对应的s值
        theta0 = float(self._theta0_s(s_shroud))
        h_max = float(self._hmax_s(s_shroud))
        t_max = float(self._tmax_s(s_shroud))

        # 获取shroud处的camber和thickness参数
        camber_ctrl = np.array([f(s_shroud) for f in self._camber_ctrl_s])
        gamma = bezier_curve(xi, camber_ctrl)
        t_ctrl = np.array([f(s_shroud) for f in self._thickness_ctrl_s])
        tau = spline_thickness(xi, self._thickness_knots_x, t_ctrl)

        # 创建两个span层的顶点
        # 第一层：shroud_radius (i=0)
        # 第二层：shroud_radius_plus (i=1)
        span_layers_vertices_upper = np.zeros((2, chord_pts, 3))
        span_layers_vertices_lower = np.zeros((2, chord_pts, 3))

        for layer_idx, radius in enumerate([self.shroud_radius, shroud_radius_plus]):
            for j, x in enumerate(xi):
                theta = self.theta_offset + theta0 + x * self.Theta
                zc = self.z0 + x * self.H - h_max * gamma[j]
                zu = zc + t_max * tau[j]
                zl = zc - t_max * tau[j]

                span_layers_vertices_upper[layer_idx, j] = [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    zu
                ]
                span_layers_vertices_lower[layer_idx, j] = [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    zl
                ]

        # 使用与to_occt_solid_loft相同的逻辑创建假想叶片的loft实体
        hyp_loft = BRepOffsetAPI_ThruSections(True, True, 1e-6)

        for i in range(2):  # 只有两个span层
            polygon_builder = BRepBuilderAPI_MakePolygon()

            # 添加上表面点（前缘到后缘）
            for j in range(chord_pts):
                x, y, z = span_layers_vertices_upper[i, j]
                polygon_builder.Add(gp_Pnt(x, y, z))

            # 添加下表面点（后缘到前缘，反向）
            for j in range(chord_pts - 1, -1, -1):
                x, y, z = span_layers_vertices_lower[i, j]
                polygon_builder.Add(gp_Pnt(x, y, z))

            # 闭合多边形
            x_first, y_first, z_first = span_layers_vertices_upper[i, 0]
            polygon_builder.Add(gp_Pnt(x_first, y_first, z_first))

            if polygon_builder.IsDone():
                wire = polygon_builder.Wire()
                hyp_loft.AddWire(wire)
            else:
                print(f"Warning: Failed to create polygon for hypothetical blade layer {i}")
        return hyp_loft

    def to_cap(self):
        """生成 hub/shroud 圆柱曲面封盖 - 直接使用布尔运算技术"""
        # 1. 生成叶片侧面实体
        blade_loft = self.to_occt_solid_loft()

        # 2. 计算叶片的轴向范围
        span_pts, chord_pts, _ = self.vertices_upper.shape
        z_min = min(np.min(self.vertices_upper[:, :, 2]), np.min(self.vertices_lower[:, :, 2]))
        z_max = max(np.max(self.vertices_upper[:, :, 2]), np.max(self.vertices_lower[:, :, 2]))

        # 3. 创建hub和shroud圆柱体（高度要足够覆盖叶片）
        cylinder_height = z_max - z_min + 0.1  # 增加10cm余量确保完全覆盖
        z_center = (z_max + z_min) / 2
        cylinder_bottom = z_center - cylinder_height / 2

        # hub圆柱体
        hub_axis = gp_Ax2(gp_Pnt(0, 0, cylinder_bottom), gp_Dir(0, 0, 1))
        hub_cylinder = BRepPrimAPI_MakeCylinder(hub_axis, self.hub_radius, cylinder_height).Shape()

        # shroud圆柱体
        shroud_axis = gp_Ax2(gp_Pnt(0, 0, cylinder_bottom), gp_Dir(0, 0, 1))
        shroud_cylinder = BRepPrimAPI_MakeCylinder(shroud_axis, self.shroud_radius, cylinder_height).Shape()

        # 4. 关键步骤：使用布尔差集生成端盖
        # hub端盖 = hub圆柱体 - 叶片侧面
        hub_cut = BRepAlgoAPI_Cut(hub_cylinder, blade_loft)
        hub_cap = hub_cut.Shape()

        # ------修改shroud端盖---------
        # 生成辅助叶片
        hyp_loft = self._summon_hyp_loft()
        hyp_loft.Build()
        blade_loft_hyp = hyp_loft.Shape()

        # shroud端盖 = shroud圆柱体 ∩ 假想叶片
        shroud_cut = BRepAlgoAPI_Common(shroud_cylinder, blade_loft_hyp)
        shroud_cap = shroud_cut.Shape()
        # ---------------------

        # 5. 验证端盖是否正确生成
        if hub_cap.IsNull():
            raise RuntimeError("Hub cap is null")
        if shroud_cap.IsNull():
            raise RuntimeError("Shroud cap is null")

        # 6. 检查端盖类型（应该是面，不是实体）
        # 检查hub_cap中的面数量
        hub_face_count = 0
        exp = TopExp_Explorer(hub_cap, TopAbs_FACE)
        while exp.More():
            hub_face_count += 1
            exp.Next()

        print(f"Hub cap contains {hub_face_count} faces")

        # 检查shroud_cap中的面数量
        shroud_face_count = 0
        exp = TopExp_Explorer(shroud_cap, TopAbs_FACE)
        while exp.More():
            shroud_face_count += 1
            exp.Next()

        print(f"Shroud cap contains {shroud_face_count} faces")

        return hub_cap, shroud_cap


if __name__ == "__main__":
    with open("./param.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    hub_radius = 0.121 / 2
    shroud_radius = 0.16 / 2
    N = 6
    H1 = (0.21 + 0.04) / 2
    H = 0.21 / 2
    z0 = 0
    Theta = np.pi / 3.5

    # =======================
    # Passage
    # =======================
    passage = AnnularSectorPassage(hub_radius, shroud_radius, z0, H1, N)
    passage_grid, passage_zrange, passage_theta_range = passage.generate_surface()

    # =======================
    # Blade
    # =======================
    blade = BladeVoid(
        config["layers_params"],
        Theta,
        H,
        z0,
        hub_radius,
        shroud_radius,
        theta_offset=0.0
    )

    blade.generate_surface(
        passage_z=passage_zrange,
        passage_theta=passage_theta_range,
        align="left"
    )

    blade.visualize(passage_grid=passage_grid)

    print("\n=== Generating blade surfaces ===")
    blade_loft = blade.to_occt_solid_loft()
    write_step_file(blade_loft, "./CQ/blade_loft.step")

    print("Blade loft null:", blade_loft.IsNull())

    # =======================
    # End caps
    # =======================
    print("\n=== Generating hub / shroud caps ===")
    hub_cap, shroud_cap = blade.to_cap()

    write_step_file(hub_cap, "./CQ/hub_cap_boolean.step")
    write_step_file(shroud_cap, "./CQ/shroud_cap_boolean.step")

    # =======================
    # Sewing → Solid
    # =======================
    print("\n=== Sewing blade ===")
    sewing = BRepBuilderAPI_Sewing(1e-5)
    sewing.Add(blade_loft)
    sewing.Add(hub_cap)
    sewing.Add(shroud_cap)
    sewing.Perform()

    closed_shell = sewing.SewedShape()
    print("Closed shell null:", closed_shell.IsNull())

    solid_maker = BRepBuilderAPI_MakeSolid(closed_shell)
    if not solid_maker.IsDone():
        raise RuntimeError("MakeSolid failed")

    complete_blade = solid_maker.Shape()
    write_step_file(complete_blade, "./CQ/blade_complete_boolean.step")

    print("Blade solid valid:",
          BRepCheck_Analyzer(complete_blade).IsValid())

    # =======================
    # Passage solid
    # =======================
    angle = 2 * np.pi / N
    annular_passage = make_annular_sector_prism_simple(
        hub_radius,
        shroud_radius,
        H1,
        angle
    )

    write_step_file(annular_passage, "./CQ/annular_passage.step")
    # =======================
    # Boolean COMMON check
    # =======================
    print("\n=== Boolean COMMON check ===")
    common = BRepAlgoAPI_Common(annular_passage, complete_blade)
    common.Build()

    print("COMMON done:", common.IsDone())
    print("COMMON null:", common.Shape().IsNull())

    if not common.Shape().IsNull():
        write_step_file(common.Shape(), "./CQ/debug_common.step")

    # =======================
    # Boolean CUT
    # =======================
    print("\n=== Boolean CUT (fluid domain) ===")
    fluid_cut = BRepAlgoAPI_Cut(annular_passage, complete_blade)
    fluid_cut.Build()

    print("CUT done:", fluid_cut.IsDone())
    fluid_solid = fluid_cut.Shape()
    print("Fluid solid null:", fluid_solid.IsNull())
    print("Fluid solid valid:",
          BRepCheck_Analyzer(fluid_solid).IsValid())

    write_step_file(fluid_solid, "./CQ/fluid_domain_complete.step")

    print("\n=== DONE ===")


