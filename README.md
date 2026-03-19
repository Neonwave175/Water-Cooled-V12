# Water-Cooled-V12
Simple system for generating optimal wings and testing them

## Tools
Below are the tools in this repo

### OptimAerofoil.py
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
Run 'bash
