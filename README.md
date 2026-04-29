# Massively Parallel MPPI Quadruped Controller (CUDA/C++)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.0+](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This repository features a high-performance **Model Predictive Path Integral (MPPI)** controller designed for quadrupedal autonomous systems. By leveraging custom CUDA kernels, the controller achieves a **1,000x speedup** over CPU-based implementations, enabling the simulation of over **1 million trajectory rollouts** in under **15ms**.

<p align="center">
  <img src="mppi_final_demo.gif" width="700" alt="MPPI Quadruped Simulation HUD">
</p>

## 🚀 Performance Summary

* **Massively Parallel:** 1,000,000+ trajectory rollouts per control cycle.
* **Hardware Acceleration:** ~1085x speedup vs. single-threaded C++ dynamics.
* **Real-Time Control:** Optimized for 50Hz–100Hz control loops on modern NVIDIA architectures (Blackwell/Thor).

| Trajectories | GPU Time (ms) | CPU Time (ms) | Speedup (x) |
| :----------- | :------------ | :------------ | :---------- |
| 1,000        | 0.16          | 14.63         | **90.8x** |
| 10,000       | 0.16          | 157.28        | **966.0x** |
| 1,000,000    | 14.89         | 16153.20      | **1085.1x** |

---

## 🧠 Technical Architecture

### 1. Stochastic Trajectory Optimization
The controller utilizes MPPI to optimize Ground Reaction Forces (GRFs). Each trajectory is weighted using a Boltzmann distribution to compute the optimal control sequence:

$$w_i = \exp\left(-\frac{1}{\lambda} (S_i - \rho)\right)$$

Where $S_i$ is the total cost of the $i$-th rollout, $\rho$ is the minimum cost for numerical stability, and $\lambda$ is the temperature parameter controlling exploration vs. exploitation.

### 2. High-Fidelity Physics Engine
* **Dynamics Model:** Implements a **Single Rigid Body Model (SRBM)** with a 12-DOF floating-base.
* **Integration:** 4th-order Runge-Kutta (RK4) integration implemented in a `__device__` function for high-precision state updates.
* **Spatial Intelligence:** A GPU-accelerated **Signed Distance Field (SDF)** provides real-time collision costs with soft-margin avoidance and hard-collision rejection.

### 3. CUDA Optimization Strategies
* **Memory Coalescing:** State structures are 16-byte aligned (`alignas(16)`) to ensure optimal L2 cache utilization and coalesced memory access.
* **Parallel Reduction:** Final control updates are aggregated using a tree-based reduction kernel with **Warp Shuffles**, bypassing expensive global memory synchronizations.
* **Asynchronous Execution:** Utilizes CUDA streams to overlap data transfer with kernel execution for minimal latency.

---

## 🛠️ Installation & Build

### Prerequisites
* **NVIDIA GPU** (Compute Capability 8.0+ recommended)
* **CUDA Toolkit 12.0+**
* **Eigen 3.4+**
* **Python 3.12+** (for visualization)

### Build Process
```powershell
# Clone the repository
git clone [https://github.com/yourusername/MPPI_Quadruped_CUDA.git](https://github.com/yourusername/MPPI_Quadruped_CUDA.git)
cd MPPI_Quadruped_CUDA

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
