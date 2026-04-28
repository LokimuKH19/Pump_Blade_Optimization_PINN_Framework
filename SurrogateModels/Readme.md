# Surrogate Modeling for the Flow Field in Axial Turbo Machineries

This filefolder facilitates an end-to-end mapping from the blade geometric definition (and operating conditions) to the flow field in the blade area directly.

## Contents

- `BladeImport.py`: Converts the parametric blade geometry into the mask of the Immerse Boundary Method (IBM), which could be transfered into both CFD and surrogate modeling workflow.
- `DataGenerator.py`: Generates CFD data in the dimensionless cylinderical coordinates via SIMPLE iteration for validating and training surrogate models.
- `PressureUpdaters.py`: Solvers of pressure correlation equation in CFD workflow.
- `Dimensionless Document.pdf`: The current plan
- `results.pptx`: some results as reminders.

------

## Updates

### Apr 28th, 2026
Finished `BladeImport.py` thoroughly.
