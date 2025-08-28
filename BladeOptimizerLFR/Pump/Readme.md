# Assembly.py Overview 🏭

This markdown file provides an introduction to the **`Assembly.py`** program, which is designed for assembling pump geometries using parameterized blade models generated from `BladeGenerator.py`. It supports rotor blades, optional vanes, and inlet/outlet diffusers, all in a flexible parametric way.

---

## Program Principle ⚙️

The program works by:

1. Loading 3D blade geometry via `Blade3D` objects.  
2. Generating a cylindrical hub body.  
3. Assembling multiple blades evenly around the cylinder.  
4. Creating inlet and outlet diffusers with configurable shapes (hemisphere or paraboloid).  
5. Combining all components into a single pump mesh.  
6. Exporting results in STL and/or VTK formats.

The program leverages **trimesh** for geometry handling and **pyvista** for export and visualization.

---

## Main Functions 📝

### Blade Assembly

- **Function:** `assemble_blades_on_cylinder(blade, n_blades, radius, height, z_base, as_solid=True)`  
- **Purpose:** Evenly distributes `n_blades` around a cylinder of given radius and height.  
- **Process:**
  - Generate the cylinder mesh.
  - Generate blade meshes (solid or surface-only).
  - Apply rotation matrices to distribute blades.
  - Concatenate cylinder and blades into a single mesh.

### Diffuser Creation

- **Function:** `create_diffuser(shape, radius_base, radius_top, height, z_base, position)`  
- **Purpose:** Parametric modeling of pump inlet/outlet shapes.  
- Hemisphere Supported 🌐:
```math
X = R \sin(\Theta) \cos(\Phi), \quad Y = R \sin(\Theta) \sin(\Phi), \quad Z = 
\begin{cases} 
-r \cos(\Theta) + z_{\text{base}} + r, & \text{bottom inlet hemisphere} \\
r \cos(\Theta) + z_{\text{base}}, & \text{top outlet hemisphere} 
\end{cases}
```
- Paraboloid Supported 🔺:
```math
X = R \cos(\Phi), \quad Y = R \sin(\Phi), \quad Z = 
\begin{cases} 
h \left(\frac{R}{R_{\text{base}}}\right)^2 + z_{\text{base}}, & \text{bottom inlet hemisphere} \\
h \left(1 - \left(\frac{R}{R_{\text{base}}}\right)^2\right) + z_{\text{base}}, & \text{top outlet hemisphere}} 
\end{cases}
```
#### Hemisphere Diffuser

- Uses spherical coordinates with azimuthal (`φ`) and polar (`θ`) subdivisions.  
- Bottom diffuser: convex downward (apex at top, base at z_base).  
- Top diffuser: convex upward (apex at top, base at z_base).

#### Paraboloid Diffuser

- Radial and azimuthal subdivisions (`r`, `φ`) generate mesh.  
- Bottom: convex downward, `Z = height * (R / radius_base)^2 + z_base`.  
- Top: convex upward, `Z = height * (1 - (R / radius_base)^2) + z_base`.

---

### Pump Assembly

- **Function:** `assemble_pump(...)`  
- **Purpose:** Assemble full pump with rotor, optional vane, inlet/outlet diffusers.  
- **Features:**
  - Validates blade span vs cylinder height.
  - Supports different numbers of blades for rotor and vane.
  - Automatically positions blades along z-axis.
  - Generates outlet shaft if needed.
  - Returns dictionary of mesh parts: `{"inlet", "rotor", "vane", "outlet", "assembly"}`

---

### Export Function

- **Function:** `export_pump(meshes, directory, export_format="both")`  
- **Purpose:** Save meshes in STL and/or VTK formats.  
- **Notes:** Exports include timestamp for versioning. Supports both PyVista PolyData and trimesh meshes.

---

## How to Use 🏃‍♂️

1. Prepare blade JSON files compatible with `Blade3D`.  
2. Adjust parameters: number of blades, cylinder radius/height, diffuser shape, outlet shaft.  
3. Run the `assemble_pump` function.  
4. Export meshes using `export_pump`.  
5. Visualize in PyVista or any STL-compatible viewer.

Example usage:

```python
meshes = assemble_pump(
    rotor_blade_file='rotor.json',
    vane_blade_file='vane.json',
    rotor_height=0.25,
    vane_height=0.25,
    n_rotor_blades=6,
    n_vane_blades=10,
    inlet_shape="hemisphere",
    outlet_shape="paraboloid",
)
export_pump(meshes, directory='./Pump', export_format="stl")
```

---
## Future Integration with Blade UI 🖥️

The goal is to connect `Assembly.py` with an **interactive Blade Parameterization UI**, allowing engineers to visually design and adjust pump components. Key points include:

- **Interactive Blade Editing:**  
  Users can define rotor and vane blade shapes, specify spanwise camber and thickness distributions, and preview changes in real-time.  

- **Diffuser Geometry Control:**  
  The UI will support parameterization of inlet and outlet diffusers:  
  - **Hemisphere 🌐** – Generated using polar (`θ`) and azimuthal (`φ`) subdivisions.
  - **Paraboloid 🔺** – Defined via radial (`r`) and azimuthal (`φ`) mesh with quadratic height function.  

- **Real-time Assembly Preview:**  
  As users adjust blade parameters or diffuser shapes, the full pump assembly can be instantly updated and visualized.  

- **Seamless Export:**  
  Meshes can be exported in STL or VTK directly from the UI, ready for CFD or CAD analysis.  

- **Parameter Synchronization:**  
  Changes made in the UI will map directly to the `Assembly.py` parameters, ensuring consistency between interactive design and programmatic assembly.
