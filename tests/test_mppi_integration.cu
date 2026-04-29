#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>

// Include the dynamics and kernels
#include "../src/dynamics.cu"

using namespace mppi;

// Helper kernel to initialize curand states
__global__ void init_curand_states(curandState* states, unsigned long long seed, int num_rollouts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < num_rollouts) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

TEST(MPPIIntegrationTest, ForwardTargetCheck) {
    const int NUM_ROLLOUTS = 512;
    const int HORIZON = 50;
    const scalar_t DT = 0.01f;
    const scalar_t NOISE_STD = 1.0f;

    // 1. Setup Initial State and Target
    State h_initial_state;
    h_initial_state.position.setZero();
    h_initial_state.position.z() = 0.3f;   // Standing height
    h_initial_state.orientation.setIdentity();
    h_initial_state.linear_velocity.setZero();
    h_initial_state.angular_velocity.setZero();

    State h_target_state = h_initial_state;
    h_target_state.position.x() = 2.0f; // Target is 2 meters straight ahead (+X)
    h_target_state.position.z() = 0.3f; // Stay at standing height

    // Robot Params (SRBM)
    RobotParams h_params;
    h_params.mass = 12.0f;
    h_params.inertia_body.setIdentity();
    h_params.inertia_inv_body.setIdentity();
    h_params.foot_pos_body.setZero();
    // Front-Left
    h_params.foot_pos_body(0, 0) = 0.2f; h_params.foot_pos_body(1, 0) = 0.1f; h_params.foot_pos_body(2, 0) = -0.3f;
    // Front-Right
    h_params.foot_pos_body(0, 1) = 0.2f; h_params.foot_pos_body(1, 1) = -0.1f; h_params.foot_pos_body(2, 1) = -0.3f;
    // Hind-Left
    h_params.foot_pos_body(0, 2) = -0.2f; h_params.foot_pos_body(1, 2) = 0.1f; h_params.foot_pos_body(2, 2) = -0.3f;
    // Hind-Right
    h_params.foot_pos_body(0, 3) = -0.2f; h_params.foot_pos_body(1, 3) = -0.1f; h_params.foot_pos_body(2, 3) = -0.3f;
    h_params.gravity << 0.0f, 0.0f, -9.81f;

    // Cost Params
    CostParams h_cost_params;
    h_cost_params.weight_position = 100.0f;
    h_cost_params.weight_velocity = 10.0f;
    h_cost_params.weight_orientation = 50.0f;
    h_cost_params.weight_angular_velocity = 10.0f;
    h_cost_params.weight_control = 0.1f;
    h_cost_params.weight_obstacle_proximity = 1000.0f;
    h_cost_params.weight_obstacle_collision = 1e6f;
    h_cost_params.obstacle_buffer_radius = 0.3f;
    h_cost_params.temperature_lambda = 1.0f;

    // Empty obstacle map for this integration test
    ObstacleMap h_obstacle_map;
    memset(&h_obstacle_map, 0, sizeof(ObstacleMap)); // Zero-init for safety
    h_obstacle_map.num_obstacles = 0;

    // Base Controls (Init to roughly gravity compensation)
    Control* h_base_controls = new Control[HORIZON];
    for (int t = 0; t < HORIZON; ++t) {
        h_base_controls[t].joint_torques.setZero();
        // roughly 30N per leg upwards (Z axis) to counteract 12kg * 9.81 = 117.72N
        h_base_controls[t].joint_torques(2) = 30.0f;  // Leg 1 Z
        h_base_controls[t].joint_torques(5) = 30.0f;  // Leg 2 Z
        h_base_controls[t].joint_torques(8) = 30.0f;  // Leg 3 Z
        h_base_controls[t].joint_torques(11) = 30.0f; // Leg 4 Z
    }

    // Allocate Device Memory
    State* d_initial_state;
    State* d_target_state;
    RobotParams* d_params;
    CostParams* d_cost_params;
    Control* d_base_controls;
    scalar_t* d_trajectory_costs;
    Control* d_first_step_noise;
    curandState* d_rng_states;
    ObstacleMap* d_obstacle_map;

    cudaMalloc(&d_initial_state, sizeof(State));
    cudaMalloc(&d_target_state, sizeof(State));
    cudaMalloc(&d_params, sizeof(RobotParams));
    cudaMalloc(&d_cost_params, sizeof(CostParams));
    cudaMalloc(&d_base_controls, HORIZON * sizeof(Control));
    cudaMalloc(&d_trajectory_costs, NUM_ROLLOUTS * sizeof(scalar_t));
    cudaMalloc(&d_first_step_noise, NUM_ROLLOUTS * sizeof(Control));
    cudaMalloc(&d_rng_states, NUM_ROLLOUTS * sizeof(curandState));
    cudaMalloc(&d_obstacle_map, sizeof(ObstacleMap));

    // Copy to Device
    cudaMemcpy(d_initial_state, &h_initial_state, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_state, &h_target_state, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &h_params, sizeof(RobotParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cost_params, &h_cost_params, sizeof(CostParams), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_controls, h_base_controls, HORIZON * sizeof(Control), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle_map, &h_obstacle_map, sizeof(ObstacleMap), cudaMemcpyHostToDevice);

    // Init Curand
    int blocks = (NUM_ROLLOUTS + 255) / 256;
    init_curand_states<<<blocks, 256>>>(d_rng_states, 1234ULL, NUM_ROLLOUTS);
    cudaDeviceSynchronize();

    // 2. Rollout Step
    mppi_rollout_kernel<<<NUM_ROLLOUTS, 1>>>(
        d_initial_state, d_base_controls, d_params, d_target_state, d_obstacle_map, d_cost_params,
        d_rng_states, d_trajectory_costs, d_first_step_noise, nullptr, HORIZON, DT, NOISE_STD
    );
    cudaDeviceSynchronize();

    // Copy costs back for Numerical Check
    scalar_t* h_trajectory_costs = new scalar_t[NUM_ROLLOUTS];
    cudaMemcpy(h_trajectory_costs, d_trajectory_costs, NUM_ROLLOUTS * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    
    scalar_t min_cost = 1e30f;
    for (int i = 0; i < NUM_ROLLOUTS; ++i) {
        if (h_trajectory_costs[i] < min_cost) {
            min_cost = h_trajectory_costs[i];
        }
    }
    
    scalar_t sum_w = 0.0f;
    for (int i = 0; i < NUM_ROLLOUTS; ++i) {
        sum_w += std::exp(-(h_trajectory_costs[i] - min_cost) / h_cost_params.temperature_lambda);
    }
    
    // Numerical Check Printout
    std::cout << "[ NUMERICAL CHECK ] Minimum Cost: " << min_cost << std::endl;
    std::cout << "[ NUMERICAL CHECK ] Sum of Weights: " << sum_w << std::endl;
    
    EXPECT_FALSE(std::isnan(min_cost));
    EXPECT_FALSE(std::isnan(sum_w));
    EXPECT_FALSE(std::isinf(sum_w));
    EXPECT_GT(sum_w, 0.0f) << "Sum of weights should be greater than zero.";

    // 3. Update Step
    size_t shared_mem_size = NUM_ROLLOUTS * sizeof(scalar_t);
    mppi_update_kernel<<<1, NUM_ROLLOUTS, shared_mem_size>>>(
        d_trajectory_costs, d_first_step_noise, d_base_controls, d_cost_params, NUM_ROLLOUTS, HORIZON
    );
    cudaDeviceSynchronize();

    // 4. Verification
    Control h_updated_control;
    // We only verify the first step control updated by MPPI
    cudaMemcpy(&h_updated_control, d_base_controls, sizeof(Control), cudaMemcpyDeviceToHost);

    // Because the target is at +X (2, 0, 0), the optimal control should push the robot forward.
    // The total force in the X direction should be positive.
    // GRFs in X direction are indices: 0, 3, 6, 9
    scalar_t total_force_x = h_updated_control.joint_torques(0) + 
                             h_updated_control.joint_torques(3) + 
                             h_updated_control.joint_torques(6) + 
                             h_updated_control.joint_torques(9);

    std::cout << "[ ASSERTION CHECK ] Total Force X: " << total_force_x << " N" << std::endl;
    
    EXPECT_GT(total_force_x, 0.0f) << "Expected positive force in +X direction to move towards target.";

    // Cleanup
    delete[] h_base_controls;
    delete[] h_trajectory_costs;
    cudaFree(d_initial_state);
    cudaFree(d_target_state);
    cudaFree(d_params);
    cudaFree(d_cost_params);
    cudaFree(d_base_controls);
    cudaFree(d_trajectory_costs);
    cudaFree(d_first_step_noise);
    cudaFree(d_rng_states);
    cudaFree(d_obstacle_map);
}
