#include <Eigen/Dense>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

// Include the dynamics and MPPI kernels
#include "dynamics.cu"

using namespace mppi;

// Helper kernel to initialize curand states
__global__ void init_curand_states(curandState *states, unsigned long long seed,
                                   int num_rollouts) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_rollouts) {
    curand_init(seed, tid, 0, &states[tid]);
  }
}

// Helper kernel to apply the optimal control to the true state and advance the
// simulation
__global__ void apply_dynamics_kernel(State *state, const Control *control,
                                      const RobotParams *params, scalar_t dt) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *state = compute_dynamics(*state, *control, *params, dt);
  }
}

// Helper kernel to shift the base control sequence for the receding horizon
__global__ void shift_controls_kernel(Control *base_controls, int horizon) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < horizon - 1) {
    base_controls[tid] = base_controls[tid + 1];
  }
  if (tid == horizon - 1) {
    base_controls[tid].joint_torques.setZero();
    base_controls[tid].joint_torques(2)  = 29.43f;
    base_controls[tid].joint_torques(5)  = 29.43f;
    base_controls[tid].joint_torques(8)  = 29.43f;
    base_controls[tid].joint_torques(11) = 29.43f;
  }
}

#include <chrono>
#include <iomanip>
#include <random>
#include <vector>

void run_benchmark() {
    std::cout << "\n=== KernelSpecialist Benchmark Utility ===" << std::endl;
    std::cout << "Benchmarking MPPI Rollout Kernel (Horizon = 50)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(15) << "Trajectories" 
              << std::setw(15) << "GPU Time (ms)" 
              << std::setw(15) << "CPU Time (ms)" 
              << "Speedup (x)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    const int HORIZON = 50;
    const scalar_t DT = 0.02f;
    const scalar_t NOISE_STD = 15.0f;
    
    // Setup dummy state and params
    State h_state;
    h_state.position.setZero();
    h_state.position.z() = 0.3f;
    h_state.orientation.setIdentity();
    h_state.linear_velocity.setZero();
    h_state.angular_velocity.setZero();
    
    State h_target = h_state;
    h_target.position.x() = 5.0f;

    RobotParams h_params;
    h_params.mass = 12.0f;
    h_params.inertia_body.setIdentity();
    h_params.inertia_inv_body.setIdentity();
    h_params.foot_pos_body.setZero();
    h_params.gravity << 0.0f, 0.0f, -9.81f;

    CostParams h_cost_params;
    h_cost_params.weight_position = 50.0f;
    h_cost_params.weight_velocity = 30.0f;
    h_cost_params.weight_orientation = 500.0f;
    h_cost_params.weight_angular_velocity = 100.0f;
    h_cost_params.weight_control = 0.01f;
    h_cost_params.weight_obstacle_proximity = 5000.0f;
    h_cost_params.weight_obstacle_collision = 1e6f;
    h_cost_params.obstacle_buffer_radius = 0.5f;
    h_cost_params.temperature_lambda = 1000.0f;

    ObstacleMap h_obstacle_map;
    h_obstacle_map.num_obstacles = 0;

    std::vector<Control> h_base_controls(HORIZON);
    for (int t = 0; t < HORIZON; ++t) {
        h_base_controls[t].joint_torques.setZero();
        h_base_controls[t].joint_torques(2) = 29.43f;
        h_base_controls[t].joint_torques(5) = 29.43f;
        h_base_controls[t].joint_torques(8) = 29.43f;
        h_base_controls[t].joint_torques(11) = 29.43f;
    }

    // Allocate device memory for parameters
    State *d_state, *d_target;
    RobotParams *d_params;
    CostParams *d_cost_params;
    ObstacleMap *d_obstacle_map;
    Control *d_base_controls;
    cudaMalloc(&d_state, sizeof(State));
    cudaMalloc(&d_target, sizeof(State));
    cudaMalloc(&d_params, sizeof(RobotParams));
    cudaMalloc(&d_cost_params, sizeof(CostParams));
    cudaMalloc(&d_obstacle_map, sizeof(ObstacleMap));
    cudaMalloc(&d_base_controls, HORIZON * sizeof(Control));

    cudaMemcpy(d_state, &h_state, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, &h_target, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &h_params, sizeof(RobotParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost_params, &h_cost_params, sizeof(CostParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle_map, &h_obstacle_map, sizeof(ObstacleMap), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_controls, h_base_controls.data(), HORIZON * sizeof(Control), cudaMemcpyHostToDevice);

    std::vector<int> rollout_counts = {100, 1000, 10000, 100000, 1000000};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int N : rollout_counts) {
        // Allocate buffers for this N
        scalar_t* d_trajectory_costs;
        Control* d_first_step_noise;
        curandState* d_rng_states;
        
        cudaMalloc(&d_trajectory_costs, N * sizeof(scalar_t));
        cudaMalloc(&d_first_step_noise, N * sizeof(Control));
        cudaMalloc(&d_rng_states, N * sizeof(curandState));

        // Init curand
        int blocks = (N + 255) / 256;
        init_curand_states<<<blocks, 256>>>(d_rng_states, 1234, N);
        cudaDeviceSynchronize();

        // GPU Benchmark
        cudaEventRecord(start);
        mppi_rollout_kernel<<<blocks, 256>>>(
            d_state, d_base_controls, d_params, d_target, d_obstacle_map, d_cost_params,
            d_rng_states, d_trajectory_costs, d_first_step_noise, nullptr,
            HORIZON, DT, NOISE_STD
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float gpu_ms = 0;
        cudaEventElapsedTime(&gpu_ms, start, stop);

        // CPU Benchmark
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        std::mt19937 gen(1234);
        std::normal_distribution<scalar_t> dist(0.0f, 1.0f);
        
        for (int k = 0; k < N; ++k) {
            State s = h_state;
            scalar_t total_cost = 0.0f;
            for (int t = 0; t < HORIZON; ++t) {
                Control u = h_base_controls[t];
                for (int i = 0; i < 12; ++i) {
                    u.joint_torques(i) += NOISE_STD * dist(gen);
                }
                s = compute_dynamics(s, u, h_params, DT);
                total_cost += compute_step_cost(s, u, h_target, h_obstacle_map, h_cost_params);
            }
            total_cost += compute_terminal_cost(s, h_target, h_obstacle_map, h_cost_params);
        }
        
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
        float cpu_ms = cpu_duration.count();

        float speedup = cpu_ms / gpu_ms;

        std::cout << std::left << std::setw(15) << N 
                  << std::setw(15) << gpu_ms 
                  << std::setw(15) << cpu_ms 
                  << speedup << "x" << std::endl;

        cudaFree(d_trajectory_costs);
        cudaFree(d_first_step_noise);
        cudaFree(d_rng_states);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_state);
    cudaFree(d_target);
    cudaFree(d_params);
    cudaFree(d_cost_params);
    cudaFree(d_obstacle_map);
    cudaFree(d_base_controls);
    
    std::cout << "--------------------------------------------------------\n" << std::endl;
}

int main() {
  // Run GPU vs CPU performance benchmark
  run_benchmark();

  const int NUM_ROLLOUTS = 512;
  const int HORIZON = 50;
  const scalar_t DT = 0.02f;        // 50Hz
  const scalar_t NOISE_STD = 15.0f; // Tighter noise for consistency
  const int NUM_ITERATIONS = 600;   // 12 seconds at 50Hz

  std::cout << "--- Starting MPPI Simulation ---" << std::endl;
  std::cout << "Iterations: " << NUM_ITERATIONS << ", Horizon: " << HORIZON
            << ", Rollouts: " << NUM_ROLLOUTS << std::endl;

  // 1. Setup Initial State and Target
  State h_state;
  h_state.position.setZero();
  h_state.position.z() = 0.3f; // Standing height (CoM above ground)
  h_state.orientation.setIdentity();
  h_state.linear_velocity.setZero();
  h_state.angular_velocity.setZero();

  State h_target_state = h_state;
  h_target_state.position.x() = 5.0f; // Target is 5 meters straight ahead (+X)
  h_target_state.position.z() = 0.3f; // Stay at standing height

  // Robot Params (SRBM)
  RobotParams h_params;
  h_params.mass = 12.0f;
  h_params.inertia_body.setIdentity();
  h_params.inertia_inv_body.setIdentity();
  h_params.foot_pos_body.setZero();
  h_params.foot_pos_body(0, 0) = 0.2f;
  h_params.foot_pos_body(1, 0) = 0.1f;
  h_params.foot_pos_body(2, 0) = -0.3f; // FL
  h_params.foot_pos_body(0, 1) = 0.2f;
  h_params.foot_pos_body(1, 1) = -0.1f;
  h_params.foot_pos_body(2, 1) = -0.3f; // FR
  h_params.foot_pos_body(0, 2) = -0.2f;
  h_params.foot_pos_body(1, 2) = 0.1f;
  h_params.foot_pos_body(2, 2) = -0.3f; // HL
  h_params.foot_pos_body(0, 3) = -0.2f;
  h_params.foot_pos_body(1, 3) = -0.1f;
  h_params.foot_pos_body(2, 3) = -0.3f; // HR
  h_params.gravity << 0.0f, 0.0f, -9.81f;

  // Cost Params
  CostParams h_cost_params;
  h_cost_params.weight_position = 50.0f; // Stronger goal attraction
  h_cost_params.weight_velocity =
      30.0f; // Moderate damping (braking via terminal cost)
  h_cost_params.weight_orientation = 500.0f;     // Moderate anti-roll
  h_cost_params.weight_angular_velocity = 100.0f; // Damping
  h_cost_params.weight_control = 0.01f;           // Cheap to use force
  h_cost_params.weight_obstacle_proximity = 5000.0f;
  h_cost_params.weight_obstacle_collision = 1e6f; // Hard barrier
  h_cost_params.obstacle_buffer_radius = 0.5f;
  h_cost_params.temperature_lambda =
      1000.0f; // Properly scaled to cost magnitude

  // Environment Obstacles
  ObstacleMap h_obstacle_map;
  memset(&h_obstacle_map, 0, sizeof(ObstacleMap)); // Zero-init all slots
  h_obstacle_map.num_obstacles = 1;
  h_obstacle_map.obstacles[0].position << 2.5f, 0.0f,
      0.0f;                                  // Placed in the path
  h_obstacle_map.obstacles[0].radius = 0.8f; // Larger radius (match test)

  // Base Controls (Gravity comp + slight forward bias)
  // Gravity: 12kg * 9.81 = 117.72N total. Split across 4 legs.
  // Front legs: slightly less Z to balance pitch torque from forward force
  // Rear legs:  slightly more Z to compensate
  // Total Z: 28.5*2 + 30.4*2 = 117.8N ≈ 117.72N (neutral buoyancy)
  Control *h_base_controls = new Control[HORIZON];
  for (int t = 0; t < HORIZON; ++t) {
    h_base_controls[t].joint_torques.setZero();
    h_base_controls[t].joint_torques(0) =
        0.0f; // No forward bias (let MPPI decide)
    h_base_controls[t].joint_torques(2) = 29.43f; // Pure gravity compensation
    h_base_controls[t].joint_torques(3) = 0.0f;
    h_base_controls[t].joint_torques(5) = 29.43f;
    h_base_controls[t].joint_torques(6) = 0.0f;
    h_base_controls[t].joint_torques(8) = 29.43f;
    h_base_controls[t].joint_torques(9) = 0.0f;
    h_base_controls[t].joint_torques(11) = 29.43f;
  }

  // Allocate Device Memory
  State *d_state;
  State *d_target_state;
  RobotParams *d_params;
  CostParams *d_cost_params;
  Control *d_base_controls;
  scalar_t *d_trajectory_costs;
  Control *d_first_step_noise;
  curandState *d_rng_states;
  ObstacleMap *d_obstacle_map;
  scalar_t *d_debug_trajectories;

  const int DEBUG_ROLLOUTS = 100;
  const int STATE_DIM = 13; // 3 pos + 4 quat + 3 lin_vel + 3 ang_vel

  cudaMalloc(&d_state, sizeof(State));
  cudaMalloc(&d_target_state, sizeof(State));
  cudaMalloc(&d_params, sizeof(RobotParams));
  cudaMalloc(&d_cost_params, sizeof(CostParams));
  cudaMalloc(&d_base_controls, HORIZON * sizeof(Control));
  cudaMalloc(&d_trajectory_costs, NUM_ROLLOUTS * sizeof(scalar_t));
  cudaMalloc(&d_first_step_noise, NUM_ROLLOUTS * sizeof(Control));
  cudaMalloc(&d_rng_states, NUM_ROLLOUTS * sizeof(curandState));
  cudaMalloc(&d_obstacle_map, sizeof(ObstacleMap));
  cudaMalloc(&d_debug_trajectories, DEBUG_ROLLOUTS * HORIZON * STATE_DIM * sizeof(scalar_t));

  // Copy Initial Data to Device
  cudaMemcpy(d_state, &h_state, sizeof(State), cudaMemcpyHostToDevice);
  cudaMemcpy(d_target_state, &h_target_state, sizeof(State),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_params, &h_params, sizeof(RobotParams), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cost_params, &h_cost_params, sizeof(CostParams),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_base_controls, h_base_controls, HORIZON * sizeof(Control),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_obstacle_map, &h_obstacle_map, sizeof(ObstacleMap),
             cudaMemcpyHostToDevice);

  // Init Curand
  int blocks = (NUM_ROLLOUTS + 255) / 256;
  init_curand_states<<<blocks, 256>>>(d_rng_states, 1234ULL, NUM_ROLLOUTS);
  cudaDeviceSynchronize();

  // Prepare CSV Log File
  std::ofstream log_file("trajectory_log.csv");
  log_file << "iteration,x,y,z,qw,qx,qy,qz";
  for (int i=0; i<4; ++i) {
      log_file << ",f" << i+1 << "x,f" << i+1 << "y,f" << i+1 << "z";
  }
  log_file << "\n";
  log_file << 0 << "," << h_state.position.x() << "," << h_state.position.y()
           << "," << h_state.position.z() << "," << h_state.orientation.w()
           << "," << h_state.orientation.x() << "," << h_state.orientation.y()
           << "," << h_state.orientation.z();
  for (int i=0; i<12; ++i) {
      log_file << ",0.0";
  }
  log_file << "\n";

  // --- Simulation Loop ---
  for (int iter = 1; iter <= NUM_ITERATIONS; ++iter) {

    // 1. Rollout Trajectories
    mppi_rollout_kernel<<<NUM_ROLLOUTS, 1>>>(
        d_state, d_base_controls, d_params, d_target_state, d_obstacle_map,
        d_cost_params, d_rng_states, d_trajectory_costs, d_first_step_noise,
        d_debug_trajectories, HORIZON, DT, NOISE_STD);
    cudaDeviceSynchronize();

    // 2. Compute Optimal Control (Updates d_base_controls[0])
    size_t shared_mem_size = NUM_ROLLOUTS * sizeof(scalar_t);
    mppi_update_kernel<<<1, NUM_ROLLOUTS, shared_mem_size>>>(
        d_trajectory_costs, d_first_step_noise, d_base_controls, d_cost_params,
        NUM_ROLLOUTS, HORIZON);
    cudaDeviceSynchronize();

    // 3. Apply the computed optimal control to the true robot state
    apply_dynamics_kernel<<<1, 1>>>(d_state, d_base_controls, d_params, DT);
    cudaDeviceSynchronize();

    // 4. Shift base control sequence for the next receding horizon iteration
    int shift_blocks = (HORIZON + 255) / 256;
    shift_controls_kernel<<<shift_blocks, 256>>>(d_base_controls, HORIZON);
    cudaDeviceSynchronize();

    // 5. Log the new state and optimal controls (GRFs)
    cudaMemcpy(&h_state, d_state, sizeof(State), cudaMemcpyDeviceToHost);
    Control h_opt_control;
    cudaMemcpy(&h_opt_control, d_base_controls, sizeof(Control), cudaMemcpyDeviceToHost);
    
    log_file << iter << "," << h_state.position.x() << ","
             << h_state.position.y() << "," << h_state.position.z() << ","
             << h_state.orientation.w() << "," << h_state.orientation.x() << ","
             << h_state.orientation.y() << "," << h_state.orientation.z();
    
    for (int i = 0; i < 12; ++i) {
        log_file << "," << h_opt_control.joint_torques(i);
    }
    log_file << "\n";

    // 6. Log the debug trajectories for visualization
    static std::ofstream debug_file("debug_rollouts.bin", std::ios::binary);
    std::vector<scalar_t> h_debug_trajectories(DEBUG_ROLLOUTS * HORIZON * STATE_DIM);
    cudaMemcpy(h_debug_trajectories.data(), d_debug_trajectories, 
               h_debug_trajectories.size() * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    debug_file.write(reinterpret_cast<const char*>(h_debug_trajectories.data()), 
                     h_debug_trajectories.size() * sizeof(scalar_t));

    if (iter % 20 == 0) {
      std::cout << "Iteration " << iter << "/" << NUM_ITERATIONS << " | Pos: ("
                << h_state.position.x() << ", " << h_state.position.y() << ", "
                << h_state.position.z() << ")" << std::endl;
    }
  }

  std::cout << "Simulation complete. Results saved to trajectory_log.csv"
            << std::endl;

  // Cleanup
  log_file.close();
  delete[] h_base_controls;
  cudaFree(d_state);
  cudaFree(d_target_state);
  cudaFree(d_params);
  cudaFree(d_cost_params);
  cudaFree(d_base_controls);
  cudaFree(d_trajectory_costs);
  cudaFree(d_first_step_noise);
  cudaFree(d_rng_states);
  cudaFree(d_obstacle_map);
  cudaFree(d_debug_trajectories);

  return 0;
}
