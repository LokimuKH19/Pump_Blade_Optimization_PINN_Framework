# Assembly.py
import os.path

import numpy as np
import trimesh
import pyvista as pv
from datetime import datetime
from BladeGenerator import Blade3D


def assemble_blades_on_cylinder(blade: Blade3D, n_blades: int, radius: float, height: float, z_base: float, as_solid=True):
    """
    Assemble multiple blades evenly around a cylindrical pump body.
    """
    # Generate cylinder body
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=128)
    cylinder.apply_translation([0, 0, z_base + height / 2.0])

    # Generate blade mesh
    blade_mesh = (
        blade._generate_solid_from_surfaces("both") if as_solid else blade.to_pyvista_mesh("both")
    )

    # Convert PyVista → trimesh
    if isinstance(blade_mesh, pv.PolyData):
        faces = blade_mesh.faces.reshape(-1, 4)[:, 1:]
        blade_mesh = trimesh.Trimesh(vertices=blade_mesh.points, faces=faces)

    # Store z-coordinates for height check
    if not hasattr(blade, "z_coords"):
        z_all = np.concatenate([blade.vertices_upper[:, 2], blade.vertices_lower[:, 2]])
        blade.z_coords = z_all

    meshes = [cylinder]

    # Compute blade z-shift so that blade is centered on cylinder
    z_blade_span = blade.z_coords.max() - blade.z_coords.min()
    if height < z_blade_span:
        raise ValueError(f"Provided cylinder height {height} is less than blade span {z_blade_span}")
    z_shift = z_base + height / 2.0 - (blade.z_coords.min() + z_blade_span / 2.0)

    for i in range(n_blades):
        angle = 2 * np.pi * i / n_blades
        rot_matrix = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
        shifted = blade_mesh.copy()
        shifted.apply_translation([0, 0, z_shift])
        shifted.apply_transform(rot_matrix)
        meshes.append(shifted)

    return trimesh.util.concatenate(meshes)


def create_diffuser(shape: str, radius_base: float, radius_top: float, height: float, z_base: float,
                    position: str = "bottom"):
    """
    Create a diffuser/cone/paraboloid for pump inlet/outlet.

    Parameters
    ----------
    shape : str
        'hemisphere', 'paraboloid'
    radius_base : float
        Base radius (bottom)
    radius_top : float
        Top radius
    height : float
        Height of the diffuser
    z_base : float
        z-coordinate of the bottom of diffuser
    position : str
        'bottom' for inlet, 'top' for outlet (affects hemisphere orientation)
    """

    if shape == "hemisphere":
        # Parametric hemisphere, cleanly generated
        n_phi = 128  # azimuth divisions
        n_theta = 64  # polar divisions

        phi = np.linspace(0, 2 * np.pi, n_phi)
        theta = np.linspace(0, np.pi / 2, n_theta)  # half sphere
        Phi, Theta = np.meshgrid(phi, theta)

        X = radius_base * np.sin(Theta) * np.cos(Phi)
        Y = radius_base * np.sin(Theta) * np.sin(Phi)

        if position == "bottom":
            # convex downward: apex at top, base at z_base
            Z = -radius_base * np.cos(Theta) + z_base + radius_base  # shift so base at z_base
        elif position == "top":
            # convex upward: apex at top, base at z_base
            Z = radius_base * np.cos(Theta) + z_base  # base at z_base
        else:
            raise ValueError("position must be 'bottom' or 'top' for hemisphere")

        verts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

        # Build faces
        faces = []
        for i in range(n_theta - 1):
            for j in range(n_phi - 1):
                v0 = i * n_phi + j
                v1 = v0 + 1
                v2 = (i + 1) * n_phi + j + 1
                v3 = (i + 1) * n_phi + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
            # wrap around for last column
            v0 = i * n_phi + (n_phi - 1)
            v1 = i * n_phi
            v2 = (i + 1) * n_phi
            v3 = (i + 1) * n_phi + (n_phi - 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))

    elif shape == "paraboloid":
        n = 128
        phi = np.linspace(0, 2 * np.pi, n)
        r = np.linspace(0, radius_base, n)
        R, Phi = np.meshgrid(r, phi)

        if position == "bottom":  # inlet
            Z = height * (R / radius_base) ** 2
            Z += z_base
        elif position == "top":  # outlet
            # Invert paraboloid so convex faces upward
            Z = height * (1 - (R / radius_base) ** 2)  # top apex at Z=z_base+height
            Z += z_base  # shift so bottom of paraboloid at z_base
        else:
            raise ValueError("position must be 'bottom' or 'top'")
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        verts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
        faces = []
        for i in range(n - 1):
            for j in range(n - 1):
                v0 = i * n + j
                v1 = v0 + 1
                v2 = (i + 1) * n + j + 1
                v3 = (i + 1) * n + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])
        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))

    else:
        raise ValueError(f"Unknown diffuser shape {shape}")
    return mesh


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
    shaft = trimesh.creation.cylinder(radius=outlet_shaft_radius, height=outlet_shaft_length)
    shaft.apply_translation([0, 0, current_z + outlet_shaft_length / 2.0])
    outlet = trimesh.util.concatenate([outlet_diffuser, shaft])

    parts = {"inlet": inlet, "rotor": rotor, "outlet": outlet}
    if vane is not None:
        parts["vane"] = vane

    assembly = trimesh.util.concatenate(parts.values())
    parts["assembly"] = assembly
    return parts


def export_pump(meshes: dict, directory: str, export_format: str = "both"):
    """
    Export pump meshes with timestamp in vtk and/or stl.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(directory):
        os.mkdir(directory)

    def to_pyvista(tri: trimesh.Trimesh) -> pv.PolyData:
        return pv.PolyData(tri.vertices, np.hstack((np.full((len(tri.faces), 1), 3), tri.faces)))

    if export_format in {"vtk", "both"}:
        for name, mesh in meshes.items():
            vtk_mesh = to_pyvista(mesh)
            vtk_mesh.save(f"{directory}/{name}_{timestamp}.vtk")

    if export_format in {"stl", "both"}:
        for name, mesh in meshes.items():
            mesh.export(f"{directory}/{name}_{timestamp}.stl")

    print(f"Exported pump parts in {export_format} format with timestamp {timestamp}")


if __name__ == "__main__":
    rotor_file = "./Blades/blade_example_hub0.121_shroud0.160_Theta0.524_H0.210_20250826_125315.json"
    vane_file = "./Blades/blade_example_hub0.121_shroud0.160_Theta0.524_H0.210_20250826_125315.json"
    meshes = assemble_pump(
        rotor_blade_file=rotor_file,
        vane_blade_file=vane_file,
        rotor_height=0.25,
        vane_height=0.25,
        n_rotor_blades=6,
        n_vane_blades=10,
        outlet_shaft_radius=0.05,
        outlet_shaft_length=0.3,
        inlet_shape="hemisphere",    # 'hemisphere', 'paraboloid'
        outlet_shape="paraboloid",
        as_solid=True,
    )
    export_pump(meshes, directory='./Pump', export_format="stl")    # vtk, stl, both
