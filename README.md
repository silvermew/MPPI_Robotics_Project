# MPPI Quadruped Controller (CUDA/C++)

<div align="center">
  <video src="mppi_pyvista_demo.mp4" width="100%" controls autoplay loop muted></video>
</div>

This project implements a high-performance **Model Predictive Path Integral (MPPI)** controller for a quadrupedal robot using a **Single Rigid Body Model (SRBM)**. The physics engine and MPPI optimization loop are written entirely in C++ and CUDA, allowing for massive parallel trajectory rollouts.

## Key Features

- **CUDA-Accelerated Dynamics:** SRBM physics integrated directly on the GPU to simulate millions of trajectories per second.
- **Drift Correction & Parking Brake:** Tuned orientation costs and distance-dependent velocity damping for precise target acquisition and stable orientation.
- **Obstacle Avoidance:** GPU-accelerated spatial mapping with configurable collision penalties and buffer radii.
- **3D Visualization:** A PyVista-based Python script that renders the quadruped, ground reaction forces, and the MPPI "ghost" exploration trajectories.

## Prerequisites

- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** (tested with v13.2+).
- **CMake** (v3.18 or higher).
- **C++ Compiler** (MSVC on Windows or GCC/Clang on Linux).
- **Python 3.12+** (for visualization).

## Build Instructions

This project uses CMake. To build the CUDA simulation and tests:

```powershell
# Clone the repository
git clone https://github.com/yourusername/MPPI_Robotics_Project.git
cd MPPI_Robotics_Project

# Configure and build (Windows/Release mode)
cmake -B build
cmake --build build --config Release
```

## Running the Simulation

Execute the main MPPI simulation. This will automatically run a brief performance benchmark before starting the receding horizon control loop.

```powershell
# Run the simulation (Windows)
.\build\Release\mppi_sim.exe
```

The simulation will generate two files in the root directory:
- `trajectory_log.csv`: Contains the robot's state and Ground Reaction Forces (GRFs) over time.
- `debug_rollouts.bin`: Contains the raw GPU rollout states for visualization.

## Visualization

A Python visualizer is provided to observe the robot's behavior and the MPPI search space. 

1. Create and activate a virtual environment (optional but recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Run the visualization script:
```powershell
python scripts/visualize_trajectory.py
```
This will read the data files and generate an `mppi_pyvista_demo.mp4` video demonstrating the robot navigating through obstacles.
