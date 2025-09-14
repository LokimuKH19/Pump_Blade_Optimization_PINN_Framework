# SymPhONIC may use these functions later

> Symmetric-based Physics Oriented Neural Integral Computation (SymPhONIC)

> A novel PINN framework dedicated to analyzing impellers

## Update Records 
### Sept.12, 2025
- Make sure the model could be generated in a watertight, closed volumn and can be directly import to the CAD/3D modeling softwares.
- All parameters could be transfered by a json file to describe a pump design. You don't have to use the stl/vtk file directly to save spaces.

## Dependencies Description
It is recommended to download "Blender" and "OpenSCAD" when using this software, and add them into Path (Environment Variable) for boolean calculation.

If you are using later version of the `trimesh` package, the "OpenSCAD" engine will no longer available. Instead, you can download `pymanifold` directly by 

```cmd
pip install pymanifold
``` 

in the `cmd` window and added a parameter `engine="manifold"` when conducting a boolean calculation, for example:

```python
tmp = trimesh.boolean.difference([fluid_tm, cutter_bottom], engine="manifold")
```

