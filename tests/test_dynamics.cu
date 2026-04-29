#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "../include/robot_types.h"

// Including the .cu file directly for testing purposes to access the __device__ function 
// and the RobotParams struct without needing a separate header definition.
#include "../src/dynamics.cu"

using namespace mppi;

/**
 * @brief A simple __global__ wrapper to launch the device-only compute_dynamics function.
 */
__global__ void run_dynamics_kernel(const State* state, const Control* control, const RobotParams* params, scalar_t dt, State* next_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *next_state = compute_dynamics(*state, *control, *params, dt);
    }
}

/**
 * @brief Verifies the SRBM dynamics output when a force is applied exclusively to the front-right foot.
 * 
 * Expectations:
 * - A forward force (+X) should cause a positive linear velocity in X.
 * - An upward force (+Z) combined with gravity should result in a net linear acceleration.
 * - The offset of the front-right foot should induce specific torques:
 *   - Negative Roll (-X) since the right side is pushed up.
 *   - Negative Pitch (-Y) since the front side is pushed up.
 *   - Positive Yaw (+Z) since the right side is pushed forward.
 */
TEST(DynamicsTest, FrontRightFootForceIntegration) {
    // 1. Initialize Host Data
    State h_state;
    h_state.position.setZero();
    h_state.orientation.setIdentity(); // w=1, x=y=z=0
    h_state.linear_velocity.setZero();
    h_state.angular_velocity.setZero();

    RobotParams h_params;
    h_params.mass = 10.0f;
    h_params.gravity << 0.0f, 0.0f, -9.81f;
    h_params.inertia_body.setIdentity();     // Identity matrix for simple torque verification
    h_params.inertia_inv_body.setIdentity();
    
    // Define Foot Positions in body frame (assuming +X is forward, +Y is left, +Z is up)
    // Front-Right (FR) foot at index 0. Right means -Y. Front means +X.
    h_params.foot_pos_body.col(0) <<  0.3f, -0.2f, -0.3f; // FR
    h_params.foot_pos_body.col(1) <<  0.3f,  0.2f, -0.3f; // FL
    h_params.foot_pos_body.col(2) << -0.3f, -0.2f, -0.3f; // RR
    h_params.foot_pos_body.col(3) << -0.3f,  0.2f, -0.3f; // RL

    Control h_control;
    h_control.joint_torques.setZero(); // Zero out all GRFs initially
    
    // Apply Force ONLY to the Front-Right foot (first 3 elements of joint_torques)
    // Applying 10 N forward (+X), 0 N lateral, 100 N upward (+Z)
    h_control.joint_torques.segment<3>(0) << 10.0f, 0.0f, 100.0f;

    scalar_t dt = 0.01f;

    // 2. Allocate and Copy to Device
    State* d_state;
    Control* d_control;
    RobotParams* d_params;
    State* d_next_state;

    cudaMalloc(&d_state, sizeof(State));
    cudaMalloc(&d_control, sizeof(Control));
    cudaMalloc(&d_params, sizeof(RobotParams));
    cudaMalloc(&d_next_state, sizeof(State));

    cudaMemcpy(d_state, &h_state, sizeof(State), cudaMemcpyHostToDevice);
    cudaMemcpy(d_control, &h_control, sizeof(Control), cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, &h_params, sizeof(RobotParams), cudaMemcpyHostToDevice);

    // 3. Launch Kernel (1 block, 1 thread)
    run_dynamics_kernel<<<1, 1>>>(d_state, d_control, d_params, dt, d_next_state);
    cudaDeviceSynchronize();

    // 4. Copy Back to Host
    State h_next_state;
    cudaMemcpy(&h_next_state, d_next_state, sizeof(State), cudaMemcpyDeviceToHost);

    cudaFree(d_state);
    cudaFree(d_control);
    cudaFree(d_params);
    cudaFree(d_next_state);

    // 5. Verify Numerical Correctness (Assertions)

    // Expected Linear Dynamics:
    // Net Force = F_FR + mass * gravity = [10, 0, 100] + [0, 0, -98.1] = [10, 0, 1.9]
    // Acceleration = Net Force / mass = [1.0, 0.0, 0.19]
    // Expected dv = a * dt = [0.01, 0.0, 0.0019]
    
    EXPECT_GT(h_next_state.linear_velocity.x(), 0.0f) << "Expected forward linear velocity (+X)";
    EXPECT_NEAR(h_next_state.linear_velocity.x(), 0.01f, 1e-5f);
    
    EXPECT_GT(h_next_state.linear_velocity.z(), 0.0f) << "Expected upward linear velocity (+Z)";
    EXPECT_NEAR(h_next_state.linear_velocity.z(), 0.0019f, 1e-5f);
    
    // Expected Angular Dynamics (Torque):
    // r x F = [0.3, -0.2, -0.3] x [10, 0, 100]
    //       = [ (-0.2)(100) - (-0.3)(0), (-0.3)(10) - (0.3)(100), (0.3)(0) - (-0.2)(10) ]
    //       = [ -20, -33, 2 ]
    // dw = I_inv * tau * dt = [ -20, -33, 2 ] * 0.01 = [ -0.2, -0.33, 0.02 ]

    EXPECT_LT(h_next_state.angular_velocity.x(), 0.0f) << "Expected negative roll velocity (right side pushed up)";
    EXPECT_LT(h_next_state.angular_velocity.y(), 0.0f) << "Expected negative pitch velocity (front side pushed up)";
    EXPECT_GT(h_next_state.angular_velocity.z(), 0.0f) << "Expected positive yaw velocity (right side pushed forward)";

    EXPECT_NEAR(h_next_state.angular_velocity.x(), -0.2f, 1e-5f);
    EXPECT_NEAR(h_next_state.angular_velocity.y(), -0.33f, 1e-5f);
    EXPECT_NEAR(h_next_state.angular_velocity.z(), 0.02f, 1e-5f);
}
