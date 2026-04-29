#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <cstring>

#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>

// Include the dynamics and MPPI kernels
#include "../src/dynamics.cu"

using namespace mppi;

// Helper kernel to initialize curand states
__global__ void init_curand_states_obs(curandState* states, unsigned long long seed, int num_rollouts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_rollouts) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Helper kernel to apply the optimal control to the true state
__global__ void apply_dynamics_kernel_obs(State* state, const Control* control, const RobotParams* params, scalar_t dt) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *state = compute_dynamics(*state, *control, *params, dt);
    }
}

// Helper kernel to shift the base control sequence
__global__ void shift_controls_kernel_obs(Control* base_controls, int horizon) {
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

/**
 * @brief Test: Place a large obstacle directly on the straight-line path to the target.
 * 
 * Scenario:
 *   - Robot starts at (0, 0, 0), target at (5, 0, 0).
 *   - A spherical obstacle at (2.5, 0, 0) with radius 0.8m blocks the direct path.
 *   - The robot must deviate in Y or Z to go around it.
 * 
 * Assertions:
 *   1. The robot reaches X > 3.0 (still converges towards goal).
 *   2. The robot's minimum distance to the obstacle center was never < obstacle radius (no collision).
 *   3. The robot deviated laterally (|Y| > 0.3 at some point) proving it went around the obstacle.
 */
TEST(ObstacleAvoidanceTest, AvoidSingleObstacle) {
    const int NUM_ROLLOUTS = 512;
    const int HORIZON = 50;
    const scalar_t DT = 0.02f; // 50Hz
    const scalar_t NOISE_STD = 20.0f;
    const int NUM_ITERATIONS = 400; // 8 seconds

    // 1. Setup Initial State and Target
    State h_state;
    h_state.position.setZero();
    h_state.position.z() = 0.3f;   // Standing height
    h_state.orientation.setIdentity();
    h_state.linear_velocity.setZero();
    h_state.angular_velocity.setZero();

    State h_target_state = h_state;
    h_target_state.position.x() = 5.0f;
    h_target_state.position.z() = 0.3f; // Stay at standing height

    // Robot Params (SRBM)
    RobotParams h_params;
    h_params.mass = 12.0f;
    h_params.inertia_body.setIdentity();
    h_params.inertia_inv_body.setIdentity();
    h_params.foot_pos_body.setZero();
    h_params.foot_pos_body(0, 0) = 0.2f; h_params.foot_pos_body(1, 0) = 0.1f; h_params.foot_pos_body(2, 0) = -0.3f;
    h_params.foot_pos_body(0, 1) = 0.2f; h_params.foot_pos_body(1, 1) = -0.1f; h_params.foot_pos_body(2, 1) = -0.3f;
    h_params.foot_pos_body(0, 2) = -0.2f; h_params.foot_pos_body(1, 2) = 0.1f; h_params.foot_pos_body(2, 2) = -0.3f;
    h_params.foot_pos_body(0, 3) = -0.2f; h_params.foot_pos_body(1, 3) = -0.1f; h_params.foot_pos_body(2, 3) = -0.3f;
    h_params.gravity << 0.0f, 0.0f, -9.81f;

    // Cost Params
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

    // Obstacle: placed directly in the robot's path
    ObstacleMap h_obstacle_map;
    memset(&h_obstacle_map, 0, sizeof(ObstacleMap));
    h_obstacle_map.num_obstacles = 1;
    h_obstacle_map.obstacles[0].position << 2.5f, 0.0f, 0.0f;
    h_obstacle_map.obstacles[0].radius = 0.8f;

    // Base Controls (Gravity comp + slight forward bias)
    Control* h_base_controls = new Control[HORIZON];
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
    State* d_state;
    State* d_target_state;
    RobotParams* d_params;
    CostParams* d_cost_params;
    Control* d_base_controls;
    scalar_t* d_trajectory_costs;
    Control* d_first_step_noise;
    curandState* d_rng_states;
    ObstacleMap* d_obstacle_map;

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
    cudaMemcpy(d_target_state, &h_target_state, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &h_params, sizeof(RobotParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost_params, &h_cost_params, sizeof(CostParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_controls, h_base_controls, HORIZON * sizeof(Control), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle_map, &h_obstacle_map, sizeof(ObstacleMap), cudaMemcpyHostToDevice);

    int blocks = (NUM_ROLLOUTS + 255) / 256;
    init_curand_states_obs<<<blocks, 256>>>(d_rng_states, 77ULL, NUM_ROLLOUTS);
    cudaDeviceSynchronize();

    size_t shared_mem_size = NUM_ROLLOUTS * sizeof(scalar_t);
    int shift_blocks = (HORIZON + 255) / 256;

    // --- Trajectory logging for obstacle analysis ---
    std::vector<Eigen::Vector3f> trajectory_log;
    trajectory_log.push_back(h_state.position);

    scalar_t min_dist_to_obstacle = 1e10f;
    scalar_t max_lateral_deviation = 0.0f;

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Rollout
        mppi_rollout_kernel<<<NUM_ROLLOUTS, 1>>>(
            d_state, d_base_controls, d_params, d_target_state, d_obstacle_map, d_cost_params,
            d_rng_states, d_trajectory_costs, d_first_step_noise, nullptr, HORIZON, DT, NOISE_STD
        );

        // Update
        mppi_update_kernel<<<1, NUM_ROLLOUTS, shared_mem_size>>>(
            d_trajectory_costs, d_first_step_noise, d_base_controls, d_cost_params, NUM_ROLLOUTS, HORIZON
        );

        // Apply dynamics
        apply_dynamics_kernel_obs<<<1, 1>>>(d_state, d_base_controls, d_params, DT);
        shift_controls_kernel_obs<<<shift_blocks, 256>>>(d_base_controls, HORIZON);
        cudaDeviceSynchronize();

        // Read back state to track trajectory
        cudaMemcpy(&h_state, d_state, sizeof(State), cudaMemcpyDeviceToHost);
        trajectory_log.push_back(h_state.position);

        // Track minimum distance to obstacle center
        Eigen::Vector3f obs_pos = h_obstacle_map.obstacles[0].position;
        scalar_t dist = (h_state.position - obs_pos).norm();
        if (dist < min_dist_to_obstacle) {
            min_dist_to_obstacle = dist;
        }

        // Track maximum lateral (Y/Z) deviation from the straight-line path
        scalar_t lateral = std::sqrt(h_state.position.y() * h_state.position.y() + 
                                     h_state.position.z() * h_state.position.z());
        if (lateral > max_lateral_deviation) {
            max_lateral_deviation = lateral;
        }
    }

    // --- Assertions ---
    std::cout << "[ OBSTACLE TEST ] Obstacle at: " << h_obstacle_map.obstacles[0].position.transpose()
              << ", radius: " << h_obstacle_map.obstacles[0].radius << std::endl;
    std::cout << "[ OBSTACLE TEST ] Final Position: " << h_state.position.transpose() << std::endl;
    std::cout << "[ OBSTACLE TEST ] Min distance to obstacle center: " << min_dist_to_obstacle << std::endl;
    std::cout << "[ OBSTACLE TEST ] Max lateral deviation: " << max_lateral_deviation << std::endl;

    // 1. Forward Progress: The robot should still move towards the target
    EXPECT_GT(h_state.position.x(), 2.0f)
        << "Robot failed to make forward progress towards the target.";

    // 2. Collision Avoidance: The robot should never have entered the obstacle
    EXPECT_GT(min_dist_to_obstacle, h_obstacle_map.obstacles[0].radius)
        << "Robot collided with the obstacle! Min distance (" << min_dist_to_obstacle 
        << ") < radius (" << h_obstacle_map.obstacles[0].radius << ")";

    // 3. Lateral Deviation: The robot should have deviated laterally to go around the obstacle
    EXPECT_GT(max_lateral_deviation, 0.3f)
        << "Robot did not deviate laterally to avoid the obstacle. Max lateral deviation: " << max_lateral_deviation;

    // 4. Stability Check: Check Pitch and Roll
    float w = h_state.orientation.w();
    float x = h_state.orientation.x();
    float y = h_state.orientation.y();
    float z = h_state.orientation.z();
    
    float roll = atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
    float pitch = asin(2.0f * (w * y - z * x));

    const float PI = 3.14159265358979323846f;
    float roll_deg = roll * 180.0f / PI;
    float pitch_deg = pitch * 180.0f / PI;

    std::cout << "[ STABILITY     ] Final Roll: " << roll_deg << " deg, Pitch: " << pitch_deg << " deg" << std::endl;

    EXPECT_LE(std::abs(roll_deg), 20.0f) << "Roll exceeded 20 degrees! Robot is unstable.";
    EXPECT_LE(std::abs(pitch_deg), 20.0f) << "Pitch exceeded 20 degrees! Robot is unstable.";

    // Cleanup
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
