# FluidSummoning.py
import numpy as np
import os
from Assembly import load_pump, fix_domain, create_diffuser
from BladeGenerator import Blade3D
import trimesh
import pyvista as pv
from datetime import datetime
import json

COLORS = ['#FFA853', '#92B8F9', '#F39EF9', '#7DDE6A']


# --- summon inlet flow domain ---
def inlet_fluid(reference, duct_radius: float = 0.18, reserve_z: float = 0.2):
    """
    Generate the inlet flow domain as a cylinder minus the pump inlet geometry.
    Ensure top surface of the result matches reference top (z1).
    """
    # compute z bounds of reference
    z0 = reference.bounds[0][2]
    z1 = reference.bounds[1][2]

    # create cylinder covering the inlet
    height = (z1 - (z0 - reserve_z))
    cylinder = trimesh.creation.cylinder(
        radius=duct_radius,
        height=height,
        sections=128,
        caps=True
    )
    cylinder.apply_translation([0, 0, z0 - reserve_z + height / 2])

    # slightly shift reference to ensure difference works
    epsilon = 1e-5
    reference_shifted = reference.copy()
    reference_shifted.apply_translation([0, 0, epsilon])

    # subtract reference from cylinder
    inlet = cylinder.difference(reference_shifted)

    # fix normals and faces
    fix_domain(inlet)

    # --- top surface compensation ---
    inlet_bounds = inlet.bounds
    inlet_z_max = inlet_bounds[1][2]
    dz_correction = z1 - inlet_z_max

    if abs(dz_correction) > 1e-10:
        inlet.apply_translation([0, 0, dz_correction])

    return inlet


# summon impeller_region, also can be used in the outlet
def impeller_region(reference: trimesh.Trimesh, duct_radius: float = 0.16):
    """
    Generate a cylindrical flow domain around a solid component (rotor, vane, or outlet)
    by subtracting the reference geometry from an enclosing cylinder.

    The resulting domain is watertight and ensures that the top and bottom surfaces
    exactly match the bounds of the reference solid.

    Parameters:
        reference: trimesh.Trimesh
            The solid geometry of the rotor, vane, or outlet to subtract.
        duct_radius: float
            The radius of the surrounding cylinder.

    Returns:
        impeller_domain: trimesh.Trimesh
            Watertight flow domain around the reference solid, suitable for CFD meshing.
    """
    # --- compute reference bounds ---
    z0 = reference.bounds[0][2]  # bottom of rotor/vane
    z1 = reference.bounds[1][2]  # top of rotor/vane
    height = z1 - z0

    # --- create outer cylinder ---
    cylinder = trimesh.creation.cylinder(
        radius=duct_radius,
        height=height,
        sections=128,
        caps=True
    )
    cylinder.apply_translation([0, 0, z0 + height / 2])  # center cylinder on rotor z-range

    # --- subtract reference rotor/vane ---
    impeller_domain = cylinder.difference(reference, engine='manifold')  # robust boolean

    # --- compensate for potential z-shift due to boolean ---
    # after boolean, re-align top and bottom surfaces
    if impeller_domain.bounds[0][2] > z0 or impeller_domain.bounds[1][2] < z1:
        # compute z translation needed
        current_z0 = impeller_domain.bounds[0][2]
        current_z1 = impeller_domain.bounds[1][2]
        z_translate = (z0 + z1) / 2 - (current_z0 + current_z1) / 2
        impeller_domain.apply_translation([0, 0, z_translate])

    # --- ensure normals and watertightness ---
    fix_domain(impeller_domain)

    return impeller_domain


# -- name each boundary--
def classify_face(center, r, z0, z1, part_name, duct_radius, hub_radius=0.05):
    """
    Classify a face based on its center coordinates for boundary naming.

    Parameters
    ----------
    center : np.ndarray
        Face center coordinates (x, y, z)
    r : float
        Radial distance from center
    z0, z1 : float
        Bottom and top z coordinates of the part
    part_name : str
        'inlet', 'rotor', 'vane', 'outlet'
    duct_radius : float
        Outer cylinder radius
    hub_radius : float
        Inner hub radius (for rotor/vane)

    Returns
    -------
    str
        Boundary name for the face
    """
    z_tol = 1e-4   # increase tolerance for boolean inaccuracies
    r_tol = 1e-4

    z = center[2]

    # Bottom/top surfaces
    if abs(z - z0) < z_tol:
        return f"{part_name}-in"
    elif abs(z - z1) < z_tol:
        return f"{part_name}-out"

    # Outer cylinder
    if abs(r - duct_radius) < r_tol:
        return f"{part_name}-shroud"

    # Inner cylinder or hub
    if part_name in ["rotor", "vane"]:
        if abs(r - hub_radius) < r_tol:
            return f"{part_name}-hub"
        else:
            return f"{part_name}-blade"
    else:  # inlet/outlet
        return f"{part_name}-hub"


# --Export Fluid Domain--
def export_fluid_vtk(fluid_dict, folder="./FluidVTK", duct_radius=0.18, hub_radius=0.05, timestamp: str = None):
    """
    Export each fluid part as a separate VTK file with boundary naming.

    Parameters
    ----------
    fluid_dict : dict
        Dictionary containing fluid parts, e.g., {'inlet': mesh, 'rotor': mesh, ...}
    folder : str
        Folder to save VTK files
    duct_radius : float
        Outer cylinder radius
    hub_radius : float
        Inner hub radius for rotor/vane
    timestamp: str
        Timestamp string, if None uses current time
    """
    os.makedirs(folder, exist_ok=True)
    zone_counter = 1
    boundary_id_map = {}

    for part_name, mesh in fluid_dict.items():
        # Convert trimesh to PyVista PolyData
        poly = pv.PolyData(
            mesh.vertices,
            np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
        )

        # Compute bounds
        z0, z1 = mesh.bounds[0][2], mesh.bounds[1][2]

        # Assign zone IDs
        cell_ids = []
        for f in poly.faces.reshape((-1, 4))[:, 1:]:
            center = poly.points[f].mean(axis=0)
            r = np.linalg.norm(center[:2])
            bname = classify_face(center, r, z0, z1, part_name, duct_radius, hub_radius)
            # Map to unique integer ID
            if bname not in boundary_id_map.values():
                boundary_id_map[zone_counter] = bname
                zone_id = zone_counter
                zone_counter += 1
            else:
                zone_id = [k for k, v in boundary_id_map.items() if v == bname][0]
            cell_ids.append(zone_id)

        poly.cell_data["BoundaryID"] = np.array(cell_ids, dtype=int)

        # Save VTK
        file_path = os.path.join(folder, f"{part_name}_FluidDomain_{timestamp}.vtk")
        poly.save(file_path)
        print(f"Saved {file_path} with integer zone IDs.")

    # Save the mapping table
    map_file = os.path.join(folder, f"BoundaryID_Map_{timestamp}.json")
    with open(map_file, "w") as f:
        json.dump(boundary_id_map, f, indent=4)
    print(f"Saved boundary ID mapping: {map_file}")


# -- Combine Each Part of the FlowField --
def combination_fluid_domain(pump_file: str, duct_radius: float = 0.18, reserve_z: float = 0.2):
    pump_model, meta_data = load_pump(pump_file)
    # Generate inlet fluid domain
    inlet_segment = inlet_fluid(
        reference=pump_model["inlet"],  # use inlet part as reference
        duct_radius=duct_radius,
        reserve_z=reserve_z
    )
    # Generate impeller fluid domain
    rotor_segment = impeller_region(
        reference=pump_model["rotor"],  # use inlet part as reference
        duct_radius=duct_radius,
    )
    vane_segment = impeller_region(
        reference=pump_model["vane"],  # use inlet part as reference
        duct_radius=duct_radius,
    )
    outlet_segment = impeller_region(
        reference=pump_model["outlet"],  # use inlet part as reference
        duct_radius=duct_radius,
    )
    fluid_domain_dict = {"inlet": inlet_segment, "rotor": rotor_segment, "vane": vane_segment, "outlet": outlet_segment}
    new_metadata = {
        "pump_file": pump_file,
        "duct_radius": duct_radius,
        "reserve_z": reserve_z
    }

    return fluid_domain_dict, new_metadata


# --export fluid domain --
def export_fluid_domain(fluid_domain: dict, export_path: str = './FluidDomain', export_mode: str = "stl",
                        metadata: dict = None):
    """
    Export fluid_domain meshes with timestamp in vtk and/or stl or just save JSON metadata.
    """

    # extract hub_radius
    _, pump_data = load_pump(metadata["pump_file"])
    with open(pump_data["rotor_blade_file"], "r") as file:
        hub_radius = json.load(file)["hub_radius"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(export_path):
        os.mkdir(export_path)

    # Save metadata JSON
    if metadata is not None:
        json_path = f"{export_path}/FluidDomain_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Pump metadata saved: {json_path}")

    # If only json required, exit early
    if export_mode == "json_only":
        return

    if export_mode in {"stl", "both"}:
        for name, mesh in fluid_domain.items():
            mesh.export(f"{export_path}/{name}_FluidDomain_{timestamp}.stl")

    if export_mode in {"vtk", "both"}:
        export_fluid_vtk(fluid_domain, export_path, metadata["duct_radius"], hub_radius, timestamp)

    print(f"✅ Exported pump parts in {export_mode} format with timestamp {timestamp}")


# -- load json_file ---
def load_fluid_domain(json_file: str = None, metadata: dict = None):
    """
    Reconstruct fluid_domain from saved JSON metadata.
    """
    if not metadata:
        with open(json_file, "r") as f:
            metadata = json.load(f)

    return combination_fluid_domain(
        pump_file=metadata["pump_file"],
        duct_radius=metadata["duct_radius"],
        reserve_z=metadata["reserve_z"]
    )


# visualize trimesh part in pump style
def visualize_part_pump_style(meshes: dict, opacity: float = 0.3):
    """
    Visualize a dictionary of parts in PyVista like the pump visualization.
    Each key is a part name, each value is a trimesh.Trimesh or PyVista PolyData.
    Parts are shown with semi-transparent surfaces and visible edges for inspection.
    """
    plotter = pv.Plotter()
    colors = {
        "inlet": COLORS[0],
        "rotor": COLORS[1],
        "vane": COLORS[2],
        "outlet": COLORS[3],
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

        plotter.add_mesh(
            poly,
            color=color,
            opacity=opacity,  # semi-transparent
            show_edges=False,  # show edges for inspection
            edge_color="black",
            label=name
        )

    plotter.add_legend()
    plotter.show()


# testing codes
if __name__ == '__main__':
    DUCT_RADIUS = 0.161  # Tip clearance should be reserved for summoning reasonable cases
    RESERVE_Z = 0.05
    pump_json = "./Pump/test_pump.json"

    Fluid_json = "./FluidDomain/Test_Fluid.json"

    # FLUID, METADATA = combination_fluid_domain(pump_json, DUCT_RADIUS, RESERVE_Z)
    export_modes = {"json_only", "stl", "vtk", "both"}
    FLUID, METADATA = load_fluid_domain(Fluid_json)

    export_fluid_domain(FLUID, "./FluidDomain", export_mode="vtk", metadata=METADATA)
    # Visualize
    visualize_part_pump_style(FLUID)
