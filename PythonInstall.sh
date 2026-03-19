#!/bin/bash

set -e

echo "Checking Python"

# Ensure python3 exists
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing"
    brew install python@3.12
fi

echo "Creating Venv"
python3.12 -m venv AeroVenv
source AeroVenv/bin/activate

echo "Upgrading pip"
python3 -m pip install --upgrade pip

echo "Installing required packages"
python3 -m pip install numpy scipy

echo "Test Packages"
python -c "import numpy, scipy; print('NumPy:', numpy.__version__, '| SciPy:', scipy.__version__)"

echo -e "Packages Succesfully Installed,\nTo open Venv, enter 'source AeroVenv/bin/activate'"
