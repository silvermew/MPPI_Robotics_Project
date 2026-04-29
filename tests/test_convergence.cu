#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <gtest/gtest.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <cmath>

// Include the dynamics and MPPI kernels
#include "../src/dynamics.cu"

using namespace mppi;

// Helper kernel to initialize curand states
__global__ void init_curand_states_conv(curandState *states,
                                        unsigned long long seed,
                                        int num_rollouts) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_rollouts) {
    curand_init(seed, tid, 0, &states[tid]);
  }
}

// Helper kernel to apply the optimal control to the true state
__global__ void apply_dynamics_kernel_conv(State *state, const Control *control,
                                           const RobotParams *params,
                                           scalar_t dt) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *state = compute_dynamics(*state, *control, *params, dt);
  }
}

// Helper kernel to shift the base control sequence
__global__ void shift_controls_kernel_conv(Control *base_controls,
                                           int horizon) {
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

TEST(MPPIConvergenceTest, FullLoopConvergence) {
  const int NUM_ROLLOUTS = 512;
  const int HORIZON = 50;
  const scalar_t DT = 0.02f;        // 50Hz
  const scalar_t NOISE_STD = 20.0f; // More power for convergence
  const int NUM_ITERATIONS = 400;   // 8 seconds

  // 1. Setup Initial State and Target
  State h_state;
  h_state.position.setZero();
  h_state.position.z() = 0.3f; // Standing height
  h_state.orientation.setIdentity();
  h_state.linear_velocity.setZero();
  h_state.angular_velocity.setZero();

  State h_target_state = h_state;
  h_target_state.position.x() = 5.0f; // Target is 5 meters straight ahead
  h_target_state.position.z() = 0.3f; // Stay at standing height

  // Robot Params (SRBM)
  RobotParams h_params;
  h_params.mass = 12.0f;
  h_params.inertia_body.setIdentity();
  h_params.inertia_inv_body.setIdentity();
  h_params.foot_pos_body.setZero();
  h_params.foot_pos_body(0, 0) = 0.2f;
  h_params.foot_pos_body(1, 0) = 0.1f;
  h_params.foot_pos_body(2, 0) = -0.3f;
  h_params.foot_pos_body(0, 1) = 0.2f;
  h_params.foot_pos_body(1, 1) = -0.1f;
  h_params.foot_pos_body(2, 1) = -0.3f;
  h_params.foot_pos_body(0, 2) = -0.2f;
  h_params.foot_pos_body(1, 2) = 0.1f;
  h_params.foot_pos_body(2, 2) = -0.3f;
  h_params.foot_pos_body(0, 3) = -0.2f;
  h_params.foot_pos_body(1, 3) = -0.1f;
  h_params.foot_pos_body(2, 3) = -0.3f;
  h_params.gravity << 0.0f, 0.0f, -9.81f;

  // Cost Params
  CostParams h_cost_params;
  h_cost_params.weight_position = 50.0f;
  h_cost_params.weight_velocity = 30.0f;
  h_cost_params.weight_orientation = 500.0f;
  h_cost_params.weight_angular_velocity = 100.0f;
  h_cost_params.weight_control = 0.01f;
  h_cost_params.weight_obstacle_proximity = 1000.0f;
  h_cost_params.weight_obstacle_collision = 1e6f;
  h_cost_params.obstacle_buffer_radius = 0.3f;
  h_cost_params.temperature_lambda = 1000.0f;

  ObstacleMap h_obstacle_map;
  memset(&h_obstacle_map, 0, sizeof(ObstacleMap)); // Zero-init for safety
  h_obstacle_map.num_obstacles = 0;

  // Base Controls (Gravity comp + slight forward bias)
  Control *h_base_controls = new Control[HORIZON];
  for (int t = 0; t < HORIZON; ++t) {
    h_base_controls[t].joint_torques.setZero();
    h_base_controls[t].joint_torques(0) = 0.0f;   // No bias
    h_base_controls[t].joint_torques(2) = 29.43f; // Gravity
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

  cudaMalloc(&d_state, sizeof(State));
  cudaMalloc(&d_target_state, sizeof(State));
  cudaMalloc(&d_params, sizeof(RobotParams));
  cudaMalloc(&d_cost_params, sizeof(CostParams));
  cudaMalloc(&d_base_controls, HORIZON * sizeof(Control));
  cudaMalloc(&d_trajectory_costs, NUM_ROLLOUTS * sizeof(scalar_t));
  cudaMalloc(&d_first_step_noise, NUM_ROLLOUTS * sizeof(Control));
  cudaMalloc(&d_rng_states, NUM_ROLLOUTS * sizeof(curandState));
  cudaMalloc(&d_obstacle_map, sizeof(ObstacleMap));

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

  int blocks = (NUM_ROLLOUTS + 255) / 256;
  init_curand_states_conv<<<blocks, 256>>>(d_rng_states, 42ULL, NUM_ROLLOUTS);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_time_ms = 0.0f;
  size_t shared_mem_size = NUM_ROLLOUTS * sizeof(scalar_t);
  int shift_blocks = (HORIZON + 255) / 256;

  for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
    cudaEventRecord(start);

    // Rollout
    mppi_rollout_kernel<<<NUM_ROLLOUTS, 1>>>(
        d_state, d_base_controls, d_params, d_target_state, d_obstacle_map,
        d_cost_params, d_rng_states, d_trajectory_costs, d_first_step_noise,
        nullptr, HORIZON, DT, NOISE_STD);

    // Update
    mppi_update_kernel<<<1, NUM_ROLLOUTS, shared_mem_size>>>(
        d_trajectory_costs, d_first_step_noise, d_base_controls, d_cost_params,
        NUM_ROLLOUTS, HORIZON);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float iter_time = 0;
    cudaEventElapsedTime(&iter_time, start, stop);
    total_time_ms += iter_time;

    // Apply dynamics and shift outside of the timing (we mainly care about MPPI
    // kernel time)
    apply_dynamics_kernel_conv<<<1, 1>>>(d_state, d_base_controls, d_params,
                                         DT);
    shift_controls_kernel_conv<<<shift_blocks, 256>>>(d_base_controls, HORIZON);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(&h_state, d_state, sizeof(State), cudaMemcpyDeviceToHost);

  // Assertions
  float avg_time_ms = total_time_ms / NUM_ITERATIONS;
  std::cout << "[ PERFORMANCE ] Average MPPI iteration time: " << avg_time_ms
            << " ms" << std::endl;
  std::cout << "[ CONVERGENCE ] Final Position: "
            << h_state.position.transpose() << std::endl;

  EXPECT_LT(avg_time_ms, 20.0f)
      << "MPPI loop must be under 20ms to run at 50Hz";
  EXPECT_GT(h_state.position.x(), 3.0f)
      << "Robot failed to converge towards the target";

  EXPECT_LT(std::abs(h_state.position.y()), 5.0f)
      << "Robot drifted too far laterally! Y = " << h_state.position.y();

  // Check Pitch and Roll (stability)
  // Extract Euler angles (Z, Y, X) -> Yaw, Pitch, Roll
  Eigen::Vector3f euler =
      h_state.orientation.toRotationMatrix().eulerAngles(2, 1, 0);
  // eulerAngles returns angles in ranges.
  // Usually, euler(1) is pitch, euler(2) is roll.
  // However, it's safer to extract from quaternion directly to avoid
  // singularity ranges
  float w = h_state.orientation.w();
  float x = h_state.orientation.x();
  float y = h_state.orientation.y();
  float z = h_state.orientation.z();

  float roll = atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
  float pitch = asin(2.0f * (w * y - z * x));

  const float PI = 3.14159265358979323846f;
  float roll_deg = roll * 180.0f / PI;
  float pitch_deg = pitch * 180.0f / PI;

  std::cout << "[ STABILITY   ] Final Roll: " << roll_deg
            << " deg, Pitch: " << pitch_deg << " deg" << std::endl;

  EXPECT_LE(std::abs(roll_deg), 20.0f)
      << "Roll exceeded 20 degrees! Robot is unstable.";
  EXPECT_LE(std::abs(pitch_deg), 20.0f)
      << "Pitch exceeded 20 degrees! Robot is unstable.";

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
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
}
