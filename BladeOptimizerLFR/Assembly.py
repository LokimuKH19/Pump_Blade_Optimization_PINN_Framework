# Assembly.py
import os.path
import json
import numpy as np
import trimesh
import pyvista as pv
from datetime import datetime
from BladeGenerator import Blade3D
import os


def assemble_blades_on_cylinder(blade: Blade3D, n_blades: int, radius: float, height: float, z_base: float,
                                as_solid=True):
    """
    Assemble multiple blades evenly around a cylindrical pump body and return a watertight solid.
    """

    def create_watertight_cylinder(radius, height, z_base=0.0):
        # 1. summon a cylinder
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=128, caps=True)
        cyl.apply_translation([0, 0, z_base + height / 2])

        # 3. correct, combine
        rotor = trimesh.util.concatenate([cyl])
        rotor.update_faces(rotor.unique_faces())
        rotor.remove_unreferenced_vertices()
        rotor.fill_holes()
        return rotor

    # --- Create watertight cylinder rotor ---
    rotor_solid = create_watertight_cylinder(radius, height, z_base)
    # check if the cylinder is watertight
    if not rotor_solid.is_watertight:
        raise ValueError("The cylinder is not watertight.")

    # Generate blade mesh
    blade_mesh = blade._generate_solid_from_surfaces("both") if as_solid else blade.to_pyvista_mesh("both")
    if isinstance(blade_mesh, pv.PolyData):
        faces = blade_mesh.faces.reshape(-1, 4)[:, 1:]
        blade_mesh = trimesh.Trimesh(vertices=blade_mesh.points, faces=faces)

    if not blade_mesh.is_watertight:
        raise ValueError("The blade is not watertight.")

    # Blade z coordinates for centering
    if not hasattr(blade, "z_coords"):
        z_all = np.concatenate([blade.vertices_upper[:, 2], blade.vertices_lower[:, 2]])
        blade.z_coords = z_all
    z_blade_span = blade.z_coords.max() - blade.z_coords.min()
    if height < z_blade_span:
        raise ValueError(f"Provided cylinder height {height} is less than blade span {z_blade_span}")
    z_shift = z_base + height / 2 - (blade.z_coords.min() + z_blade_span / 2)

    # Place blades around cylinder
    all_meshes = [rotor_solid]  # 存放所有网格
    for i in range(n_blades):
        angle = 2 * np.pi * i / n_blades
        rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        shifted = blade_mesh.copy()
        shifted.apply_translation([0, 0, z_shift])
        shifted.apply_transform(rot_matrix)
        all_meshes.append(shifted)

    rotor_solid = trimesh.util.concatenate(all_meshes)

    rotor_solid.update_faces(rotor_solid.nondegenerate_faces())
    rotor_solid.update_faces(rotor_solid.unique_faces())
    rotor_solid.remove_unreferenced_vertices()
    rotor_solid.fill_holes()
    if not rotor_solid.is_watertight:
        raise ValueError("The impeller is not watertight.")
    return rotor_solid


# diffuser construction
def create_paraboloid_solid(radius_base: float, height: float, z_base: float,
                            position="bottom", top_z=None, n_radial=128, n_angular=128):
    """
    Create watertight paraboloid solid with base disk at top (bottom diffuser) or bottom (top diffuser).
    """
    r = np.linspace(0, radius_base, n_radial)
    phi = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    R, Phi = np.meshgrid(r, phi)

    if position == "bottom":
        Z = height * (R / radius_base) ** 2 + z_base
        disk_z = top_z if top_z is not None else z_base + height
    else:
        Z = height * (1 - (R / radius_base) ** 2) + z_base
        disk_z = top_z if top_z is not None else z_base

    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)
    verts_side = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    faces_side = []
    for i in range(n_angular):
        for j in range(n_radial - 1):
            i_next = (i + 1) % n_angular
            v0 = i * n_radial + j
            v1 = i * n_radial + j + 1
            v2 = i_next * n_radial + j + 1
            v3 = i_next * n_radial + j
            faces_side.append([v0, v1, v2])
            faces_side.append([v0, v2, v3])

    # Add top/bottom disk
    angles = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    circle_verts = np.stack([radius_base * np.cos(angles), radius_base * np.sin(angles),
                             np.full_like(angles, disk_z)], axis=-1)
    center_vert = np.array([[0, 0, disk_z]])
    verts = np.vstack([verts_side, circle_verts, center_vert])
    center_idx = len(verts) - 1
    circle_start = len(verts_side)

    faces_disk = []
    for i in range(n_angular):
        v0 = circle_start + i
        v1 = circle_start + (i + 1) % n_angular
        faces_disk.append([v0, v1, center_idx])

    faces = np.vstack([faces_side, faces_disk])
    solid = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return solid


def create_hemisphere_solid(radius: float, z_base: float, position="bottom", top_z=None, n_phi=128, n_theta=64):
    """
    Create watertight hemisphere solid with base disk at top (bottom diffuser) or bottom (top diffuser).
    """
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0, np.pi / 2, n_theta)
    Phi, Theta = np.meshgrid(phi, theta)

    X = radius * np.sin(Theta) * np.cos(Phi)
    Y = radius * np.sin(Theta) * np.sin(Phi)
    if position == "bottom":
        Z = -radius * np.cos(Theta) + z_base + radius
        disk_z = top_z if top_z is not None else z_base + radius
    else:
        Z = radius * np.cos(Theta) + z_base
        disk_z = top_z if top_z is not None else z_base

    verts_side = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    faces_side = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            j_next = (j + 1) % n_phi
            v0 = i * n_phi + j
            v1 = i * n_phi + j_next
            v2 = (i + 1) * n_phi + j_next
            v3 = (i + 1) * n_phi + j
            faces_side.append([v0, v1, v2])
            faces_side.append([v0, v2, v3])

    # Add disk
    angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    circle_verts = np.stack([radius * np.cos(angles), radius * np.sin(angles),
                             np.full_like(angles, disk_z)], axis=-1)
    center_vert = np.array([[0, 0, disk_z]])
    verts = np.vstack([verts_side, circle_verts, center_vert])
    center_idx = len(verts) - 1
    circle_start = len(verts_side)

    faces_disk = []
    for i in range(n_phi):
        v0 = circle_start + i
        v1 = circle_start + (i + 1) % n_phi
        faces_disk.append([v0, v1, center_idx])

    faces = np.vstack([faces_side, faces_disk])
    solid = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return solid


def create_diffuser(shape: str, radius_base: float, radius_top: float, height: float, z_base: float,
                    position: str = "bottom"):
    if shape == "hemisphere":
        return create_hemisphere_solid(radius_base, z_base, position)
    elif shape == "paraboloid":
        return create_paraboloid_solid(radius_base, height, z_base, position)
    else:
        raise ValueError(f"Unknown diffuser shape {shape}")


def assemble_pump(
    rotor_blade_file: str,
    vane_blade_file: str = None,
    rotor_height: float = 0.2,
    vane_height: float = 0.2,
    n_rotor_blades: int = 12,
    n_vane_blades: int = 12,
    outlet_shaft_radius: float = 0.05,
    outlet_shaft_length: float = 0.2,
    inlet_shape: str = "hemisphere",
    outlet_shape: str = "hemisphere",
    as_solid: bool = True,
):
    """
    Assemble a pump with inlet, rotor, vane, and outlet. Inlet/outlet shapes configurable.
    """
    # Load rotor blade
    rotor_blade = Blade3D.load_metadata(rotor_blade_file)
    rotor_blade.generate_surface(points_per_chord=300)

    hub_radius = rotor_blade.hub_radius
    rotor_span = rotor_blade.vertices_upper[:, 2].max() - rotor_blade.vertices_lower[:, 2].min()
    if rotor_height < rotor_span:
        raise ValueError(f"Rotor height {rotor_height} < blade span {rotor_span}")

    # Inlet diffuser
    inlet = create_diffuser(inlet_shape, hub_radius, hub_radius, hub_radius, z_base=-hub_radius, position='bottom')

    # Rotor
    rotor = assemble_blades_on_cylinder(
        blade=rotor_blade,
        n_blades=n_rotor_blades,
        radius=hub_radius,
        height=rotor_height,
        z_base=0.0,
        as_solid=as_solid,
    )
    current_z = rotor_height

    # Vane (optional)
    vane = None
    if vane_blade_file is not None:
        vane_blade = Blade3D.load_metadata(vane_blade_file)
        vane_blade.generate_surface(points_per_chord=300)
        if not np.isclose(vane_blade.hub_radius, hub_radius, atol=1e-6):
            raise ValueError("Vane hub radius must equal rotor hub radius!")
        vane_span = vane_blade.vertices_upper[:, 2].max() - vane_blade.vertices_lower[:, 2].min()
        if vane_height < vane_span:
            raise ValueError(f"Vane height {vane_height} < blade span {vane_span}")
        vane = assemble_blades_on_cylinder(
            blade=vane_blade,
            n_blades=n_vane_blades,
            radius=hub_radius,
            height=vane_height,
            z_base=current_z,
            as_solid=as_solid,
        )
        current_z += vane_height

    # Outlet diffuser
    if outlet_shaft_radius > hub_radius:
        raise ValueError("Outlet shaft radius must not exceed hub radius!")
    outlet_diffuser = create_diffuser(outlet_shape, hub_radius, outlet_shaft_radius, hub_radius, z_base=current_z, position='top')
    shaft = trimesh.creation.cylinder(radius=outlet_shaft_radius, height=outlet_shaft_length, caps=True)
    shaft.apply_translation([0, 0, current_z + outlet_shaft_length / 2.0])
    outlet = trimesh.util.concatenate([outlet_diffuser, shaft])

    # make inlet and outlet correct
    for domain in [inlet, outlet]:
        domain.fix_normals()
        domain.update_faces(domain.unique_faces())
        domain.update_faces(domain.nondegenerate_faces())
        domain.fill_holes()
        if domain.volume < 0:
            domain.invert()

    parts = {"inlet": inlet, "rotor": rotor, "outlet": outlet}
    print("Inlet domain watertight:", inlet.is_watertight)
    print("Inlet volume:", inlet.volume)
    print("Rotor domain watertight:", rotor.is_watertight)
    print("Rotor volume:", rotor.volume)
    print("Outlet domain watertight:", outlet.is_watertight)
    print("Outlet volume:", outlet.volume)
    if vane is not None:
        parts["vane"] = vane
        print("Vane domain watertight:", vane.is_watertight)
        print("Vane volume:", vane.volume)

    assembly = trimesh.util.concatenate(parts.values())
    parts["assembly"] = assembly
    return parts


def export_pump(meshes: dict, directory: str, export_format: str = "both", metadata: dict = None):
    """
    Export pump meshes with timestamp in vtk and/or stl or just save JSON metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Save metadata JSON
    if metadata is not None:
        json_path = f"{directory}/pump_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Pump metadata saved: {json_path}")

    # If only json required, exit early
    if export_format == "json_only":
        return

    def to_pyvista(tri: trimesh.Trimesh) -> pv.PolyData:
        return pv.PolyData(tri.vertices, np.hstack((np.full((len(tri.faces), 1), 3), tri.faces)))

    if export_format in {"vtk", "both"}:
        for name, mesh in meshes.items():
            vtk_mesh = to_pyvista(mesh)
            vtk_mesh.save(f"{directory}/{name}_{timestamp}.vtk")

    if export_format in {"stl", "both"}:
        for name, mesh in meshes.items():
            mesh.export(f"{directory}/{name}_{timestamp}.stl")

    print(f"✅ Exported pump parts in {export_format} format with timestamp {timestamp}")


def load_pump(json_file: str = None, metadata: dict = None):
    """
    Reconstruct pump from saved JSON metadata.
    """
    if not metadata:
        with open(json_file, "r") as f:
            metadata = json.load(f)

    return assemble_pump(
        rotor_blade_file=metadata["rotor_blade_file"],
        vane_blade_file=metadata.get("vane_blade_file"),
        rotor_height=metadata["rotor_height"],
        vane_height=metadata["vane_height"],
        n_rotor_blades=metadata["n_rotor_blades"],
        n_vane_blades=metadata["n_vane_blades"],
        outlet_shaft_radius=metadata["outlet_shaft_radius"],
        outlet_shaft_length=metadata["outlet_shaft_length"],
        inlet_shape=metadata["inlet_shape"],
        outlet_shape=metadata["outlet_shape"],
        as_solid=metadata["as_solid"],
    ), metadata


def visualize_pump(meshes: dict):
    """
    Visualize pump in PyVista with legend for each part.
    """
    plotter = pv.Plotter()
    colors = {
        "inlet": "lightblue",
        "rotor": "orange",
        "vane": "green",
        "outlet": "red",
        "assembly": "gray",
    }
    for name, mesh in meshes.items():
        if name == "assembly":
            continue
        color = colors.get(name, "white")
        if isinstance(mesh, trimesh.Trimesh):
            poly = pv.PolyData(
                mesh.vertices,
                np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
            )
        else:
            poly = mesh
        plotter.add_mesh(poly, color=color, show_edges=False, opacity=1.0, label=name)

    # automatically add the legend
    plotter.add_legend()
    plotter.show()


def create_fluid_domain_vtk(pump_json: str, output_dir: str,
                            inlet_extra: float = 0.2, outlet_extra: float = 0,
                            duct_diameter: float = 0.16):
    """
    Create fluid domain volumes for a pump and save each segment as a VTK file with
    surfaces labeled according to inlet/rotor/vane/outlet naming scheme.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load pump meshes and metadata
    meshes, metadata = load_pump(json_file=pump_json)
    assembly = meshes["assembly"]

    # Overall bounds
    zmin, zmax = assembly.bounds[0][2], assembly.bounds[1][2]
    radius = duct_diameter / 2.0

    # Create duct surrounding pump
    height = (zmax - zmin) + inlet_extra + outlet_extra
    duct = trimesh.creation.cylinder(radius=radius, height=height, sections=128, caps=True)
    duct.apply_translation([0, 0, zmin - inlet_extra + height/2])

    # Fluid volume = duct minus pump
    print("Duct domain watertight:", duct.is_watertight)
    print("Duct volume:", duct.volume)
    print("Assembly domain watertight:", assembly.is_watertight)
    print("Assembly volume:", assembly.volume)
    fluid_tm = duct.difference(assembly, engine="blender")

    # Determine rotor/vane bounds
    rotor_zmin, rotor_zmax = meshes["rotor"].bounds[:, 2]
    vane = meshes.get("vane", None)
    if vane:
        vane_zmin, vane_zmax = vane.bounds[:, 2]

    # Helper: create a z-slice of the fluid
    def slice_fluid(z0, z1):
        # Extrude infinite planes as large boxes for slicing
        cutter_bottom = trimesh.creation.box([radius*2, radius*2, 0.01])
        cutter_bottom.apply_translation([0,0,z0])
        cutter_top = trimesh.creation.box([radius*2, radius*2, 0.01])
        cutter_top.apply_translation([0,0,z1])
        # Cut with difference
        tmp = fluid_tm.difference(cutter_bottom, engine="blender")
        tmp = tmp.difference(cutter_top, engine="blender")
        return tmp

    # Function to convert trimesh volume to pyvista PolyData with cell data marking
    def trimesh_to_pvvolume(tmesh, labels):
        """
        tmesh: trimesh.Trimesh closed volume
        labels: dict mapping face index to label string
        """
        faces = tmesh.faces
        verts = tmesh.vertices
        pv_mesh = pv.PolyData(verts, np.hstack((np.full((len(faces),1),3), faces)))
        # create cell data for labels
        cell_labels = np.empty(len(faces), dtype=object)
        for i, f in enumerate(faces):
            cell_labels[i] = labels.get(i, "wall")  # default to wall
        pv_mesh.cell_data["surface_label"] = cell_labels
        return pv_mesh

    # Create four segments
    segments = {}

    # ------------------ INLET ------------------
    inlet_tm = slice_fluid(zmin - inlet_extra, rotor_zmin)
    inlet_labels = {}
    # Label faces by z/r
    for i, f in enumerate(inlet_tm.faces):
        verts = inlet_tm.vertices[f]
        if np.allclose(verts[:,2], zmin - inlet_extra, atol=1e-5):
            inlet_labels[i] = "pump_inlet"
        elif np.allclose(verts[:,2], rotor_zmin, atol=1e-5):
            inlet_labels[i] = "ir_interface"
        elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), radius, atol=1e-5):
            inlet_labels[i] = "inlet_shroud"
        else:
            inlet_labels[i] = "inlet_hub"

    pv_inlet = trimesh_to_pvvolume(inlet_tm, inlet_labels)
    pv_inlet.save(os.path.join(output_dir,"inlet.vtk"))
    segments["inlet"] = pv_inlet

    # ------------------ ROTOR ------------------
    rotor_tm = slice_fluid(rotor_zmin, rotor_zmax)
    rotor_metadata = Blade3D.load_metadata(metadata["rotor_blade_file"])
    rotor_hub_r = rotor_metadata.hub_radius
    rotor_labels = {}
    for i, f in enumerate(rotor_tm.faces):
        verts = rotor_tm.vertices[f]
        if np.allclose(verts[:,2], rotor_zmin, atol=1e-5):
            rotor_labels[i] = "ri_interface"
        elif np.allclose(verts[:,2], rotor_zmax, atol=1e-5):
            rotor_labels[i] = "rv_interface"
        elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), radius, atol=1e-5):
            rotor_labels[i] = "rotor_shroud"
        elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), rotor_hub_r, atol=1e-5):
            rotor_labels[i] = "rotor_hub"
        else:
            rotor_labels[i] = "rotor_blade"

    pv_rotor = trimesh_to_pvvolume(rotor_tm, rotor_labels)
    pv_rotor.save(os.path.join(output_dir,"rotor.vtk"))
    segments["rotor"] = pv_rotor

    # ------------------ VANE ------------------
    if vane:
        vane_tm = slice_fluid(vane_zmin, vane_zmax)
        vane_metadata = Blade3D.load_metadata(metadata["vane_blade_file"])
        vane_hub_r = vane_metadata.hub_radius
        vane_labels = {}
        for i, f in enumerate(vane_tm.faces):
            verts = vane_tm.vertices[f]
            if np.allclose(verts[:,2], vane_zmin, atol=1e-5):
                vane_labels[i] = "vr_interface"
            elif np.allclose(verts[:,2], vane_zmax, atol=1e-5):
                vane_labels[i] = "vo_interface"
            elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), radius, atol=1e-5):
                vane_labels[i] = "vane_shroud"
            elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), vane_hub_r, atol=1e-5):
                vane_labels[i] = "vane_hub"
            else:
                vane_labels[i] = "vane_blade"

        pv_vane = trimesh_to_pvvolume(vane_tm, vane_labels)
        pv_vane.save(os.path.join(output_dir, "vane.vtk"))
        segments["vane"] = pv_vane

    # ------------------ OUTLET ------------------
    outlet_zmin = vane_zmax if vane else rotor_zmax
    outlet_zmax = zmax + outlet_extra
    outlet_tm = slice_fluid(outlet_zmin, outlet_zmax)
    outlet_labels = {}
    for i, f in enumerate(outlet_tm.faces):
        verts = outlet_tm.vertices[f]
        if np.allclose(verts[:,2], outlet_zmin, atol=1e-5):
            outlet_labels[i] = "ov_interface"
        elif np.allclose(verts[:,2], outlet_zmax, atol=1e-5):
            outlet_labels[i] = "pump_outlet"
        elif np.allclose(np.linalg.norm(verts[:,:2], axis=1), radius, atol=1e-5):
            outlet_labels[i] = "outlet_shroud"
        else:
            outlet_labels[i] = "outlet_hub"

    pv_outlet = trimesh_to_pvvolume(outlet_tm, outlet_labels)
    pv_outlet.save(os.path.join(output_dir,"outlet.vtk"))
    segments["outlet"] = pv_outlet

    return segments, metadata


def visualize_fluid_domain(segments: dict):
    """
    Visualize fluid domain segments with colors and legend according to surface labels.

    Parameters
    ----------
    segments : dict
        Output from create_fluid_domain_vtk, keys: "inlet", "rotor", "vane", "outlet"
        Each value is a PyVista PolyData with cell_data["surface_label"]
    """
    plotter = pv.Plotter()
    # Define a color map for surface labels
    color_map = {
        "pump_inlet": "lightblue",
        "ir_interface": "blue",
        "inlet_shroud": "cyan",
        "inlet_hub": "navy",
        "ri_interface": "orange",
        "rv_interface": "red",
        "rotor_shroud": "gold",
        "rotor_hub": "brown",
        "rotor_blade": "darkorange",
        "vr_interface": "green",
        "vo_interface": "lime",
        "vane_shroud": "teal",
        "vane_hub": "darkgreen",
        "vane_blade": "forestgreen",
        "ov_interface": "magenta",
        "pump_outlet": "purple",
        "outlet_shroud": "pink",
        "outlet_hub": "deeppink",
        "wall": "gray"
    }

    for seg_name, pv_mesh in segments.items():
        if pv_mesh is None:
            continue
        labels = pv_mesh.cell_data["surface_label"]
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            submesh = pv_mesh.extract_cells(mask)
            plotter.add_mesh(submesh, color=color_map.get(label, "white"), show_edges=False, opacity=1.0, label=label)

    plotter.add_legend()
    plotter.show()


if __name__ == "__main__":
    rotor_file = "./Blades/blade_example_hub0.121_shroud0.160_Theta1.047_H0.210_20250909_103633.json"
    vane_file = "./Blades/blade_example_hub0.121_shroud0.160_Theta0.524_H0.210_20250826_125315.json"
    # summon metadata of the pump
    '''
    pump_metadata = {
        "rotor_blade_file": rotor_file,
        "vane_blade_file": vane_file,
        "rotor_height": 0.25,
        "vane_height": 0.25,
        "n_rotor_blades": 6,
        "n_vane_blades": 10,
        "outlet_shaft_radius": 0.05,
        "outlet_shaft_length": 0.3,
        "inlet_shape": "paraboloid",
        "outlet_shape": "hemisphere",
        "as_solid": True,
    }
    '''
    pump_json = "./Pump/test_pump.json"
    with open(pump_json, "r") as f:
        metadata = json.load(f)
    # todo Don't delete the notations here
    # meshes = load_pump(metadata=pump_metadata)    # metadata (priority) or filepath
    meshes, metadata = load_pump(metadata=metadata)
    # export_format = vtk, stl, both, json_only
    # export_pump(meshes, directory='./Pump', export_format="json_only", metadata=metadata)
    visualize_pump(meshes)
    segments, metadata = create_fluid_domain_vtk(pump_json, output_dir="./FluidDomain",
                                                 inlet_extra=0.2, outlet_extra=0.1, duct_diameter=0.16)
    visualize_fluid_domain(segments)
