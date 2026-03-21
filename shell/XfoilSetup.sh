#!/bin/bash

# Stop if anything fails
set -e

echo "Installing dependencies (macOS with Homebrew)"
brew install cmake gcc

echo "Cloning XFOIL repository"
git clone https://github.com/RobotLocomotion/xfoil.git XfoilRun

echo "Entering directory"
cd XfoilRun

echo "Creating build folder"
mkdir -p build
cd build

echo "Running CMake"
cmake ..

echo "Building"
make -j$(sysctl -n hw.ncpu)

echo "Build complete!"

echo "Open XFOIL"
cd src
./xfoil
