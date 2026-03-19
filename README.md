# Water-Cooled-V12
Simple system for generating optimal wings and testing them

## Tools
Below are the tools in this repo

### OptimAerofoilmake.py
A Python-based tool to optimize NACA 4-digit airfoils for maximum lift-to-drag ratio (L/D) using XFOIL and differential evolution.
#### What It Does
* Generates aerofoil geometry
* Runs Xfoil Simulations
* Evaluates performance across Reynolds numbers
* Finds the best aerofoil for the given re

#### Features
* Optimizes NACA 4-digit parameters (m, p, t)
* Multi-Reynolds evaluation
* Automatic fluke detection & confirmation
* Uses SciPy differential evolution
* Colored terminal output for readability

####  Python Requirements
##### Python part
Run `bash PythonInstall.sh`, This installs all required python packages and creates the venv
To activate the venv type `source AeroVenv/bin/activate`
##### Xfoil
Run `bash XfoilSetup.sh` Then move the file xfoil to `/usr/local/bin/xfoil`

### NACAToSU2.py
A Python tool to generate, visualise, and export a CFD-ready C-type mesh around any NACA 4-digit airfoil.
#### What It Does
* Creates SU2 files from NACA 4 codes
