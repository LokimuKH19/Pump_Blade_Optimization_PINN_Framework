# Surrogate Modeling for the Flow Field in Axial Turbo Machineries

This filefolder facilitates an end-to-end mapping from the blade geometric definition (and operating conditions) to the flow field in the blade area directly.

## Contents

- `BladeImport.py`: Converts the parametric blade geometry into the mask of the Immerse Boundary Method (IBM), which could be transfered into both CFD and surrogate modeling workflow.
- `DataGenerator.py`: Generates CFD data in the dimensionless cylinderical coordinates via SIMPLE iteration for validating and training surrogate models.
- `PressureUpdaters.py`: Solvers of pressure correlation equation in CFD workflow.
- `SurrogateModeling.py`: The main process of the current modeling methodology.
- `SurrogateModelingUtils.py`: Necessary dependencies for `SurrogateModeling.py`.
- `NeuralOperators.py`: Operator learning toolkit. 
- `Dimensionless Document.pdf`: The current plan for data generator.
  
------

## Updates

### Apr 28th, 2026
Finished `BladeImport.py` thoroughly.

### May 2nd, 2026
Updated `SurrogateModeling.py` and its dependencies `SurrogateModelingUtils.py`, facilitating the whole process of flow field surrogate modeling of the axial pump at certain operating conditions using a 2D-FNO([CFNO](https://github.com/LokimuKH19/SymPhONIC)) which shares parameters for each $(\Theta, Z)$ cylinderical layer. It supports model training, saving, reading and other CFD pre-post processes.

### May 4th, 2026
Updated `SurrogateModeling.py`. The 2 constants in the IBM mask are set as learnable parameters now.
