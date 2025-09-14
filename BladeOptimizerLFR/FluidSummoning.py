# FluidSummoning.py
import numpy as np
import os
from Assembly import load_pump, fix_domain, create_diffuser
from BladeGenerator import Blade3D
import trimesh
import pyvista as pv


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
            opacity=opacity,       # semi-transparent
            show_edges=False,       # show edges for inspection
            edge_color="black",
            label=name
        )

    plotter.add_legend()
    plotter.show()


# testing codes
if __name__ == '__main__':
    DUCT_RADIUS = 0.161    # Tip clearance should be reserved for summoning reasonable cases
    RESERVE_Z = 0.05
    pump_json = "./Pump/test_pump.json"

    # Load pump assembly
    pump_model, meta_data = load_pump(pump_json)

    # Generate inlet fluid domain
    inlet_segment = inlet_fluid(
        reference=pump_model["inlet"],  # use inlet part as reference
        duct_radius=DUCT_RADIUS,
        reserve_z=RESERVE_Z
    )
    print(inlet_segment.is_watertight)
    # Generate impeller fluid domain
    rotor_segment = impeller_region(
        reference=pump_model["rotor"],  # use inlet part as reference
        duct_radius=DUCT_RADIUS,
    )
    print(rotor_segment.is_watertight)
    vane_segment = impeller_region(
        reference=pump_model["vane"],  # use inlet part as reference
        duct_radius=DUCT_RADIUS,
    )
    print(vane_segment.is_watertight)
    # rotor_segment.export("rotor.stl")
    # vane_segment.export("vane.stl")
    outlet_segment = impeller_region(
        reference=pump_model["outlet"],  # use inlet part as reference
        duct_radius=DUCT_RADIUS,
    )
    print(outlet_segment.is_watertight)
    outlet_segment.export("outlet.stl")
    # Visualize
    visualize_part_pump_style({"inlet": inlet_segment, "rotor": rotor_segment,
                               "vane": vane_segment, "outlet": outlet_segment})
