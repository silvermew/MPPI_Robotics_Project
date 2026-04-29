import numpy as np
import matplotlib.pyplot as plt
import time
import os

def simulate_quadrotor_reference(steps=1000, dt=0.01):
    """
    A pure Python reference implementation of the quadrotor's vertical dynamics.
    Used by the Verifier to benchmark Python CPU performance vs expected GPU throughput.
    """
    z = 0.0
    vz = 0.0
    mass = 1.0
    gravity = 9.81
    
    # 10 N total thrust (slightly higher than 9.81 N gravity) -> Net acceleration upwards
    total_thrust = 10.0 
    
    trajectory = []
    
    print("Starting Python CPU Reference Benchmark...")
    start_time = time.perf_counter()
    
    for _ in range(steps):
        # a = (F / m) - g
        accel = (total_thrust / mass) - gravity
        vz += accel * dt
        z += vz * dt
        trajectory.append(z)
        
    end_time = time.perf_counter()
    
    elapsed_ms = (end_time - start_time) * 1000
    print(f"✅ Python CPU Simulation (1 trajectory, {steps} steps) completed in {elapsed_ms:.3f} ms")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(steps) * dt, trajectory, label='Z-Altitude', color='#2ca02c')
    plt.title("Quadrotor Altitude over Time (Python Reference)", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Altitude (m)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_plot.png")
    plt.savefig(plot_path)
    print(f"📊 Visualization saved to: {plot_path}")

if __name__ == "__main__":
    simulate_quadrotor_reference()
