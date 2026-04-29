#ifndef MPPI_COST_FUNCTION_H
#define MPPI_COST_FUNCTION_H

#include "robot_types.h"
#include <Eigen/Dense>
#include <cmath>

namespace mppi {

/**
 * @brief Weights for the Model Predictive Path Integral (MPPI) cost function.
 * 
 * Defines how much to penalize deviations from the target state and control effort.
 * Memory is 16-byte aligned to ensure efficient, coalesced memory access on GPU.
 */
struct alignas(16) CostParams {
    scalar_t weight_position;     // Penalty for Cartesian position error
    scalar_t weight_velocity;     // Penalty for linear velocity error
    scalar_t weight_orientation;  // Penalty for quaternion orientation error
    scalar_t weight_angular_velocity; // Penalty for angular velocity (spinning)
    scalar_t weight_control;      // Penalty for control magnitude (Ground Reaction Forces)
    scalar_t weight_obstacle_proximity; // Soft penalty for being near an obstacle
    scalar_t weight_obstacle_collision; // Massive penalty for colliding with an obstacle
    scalar_t obstacle_buffer_radius;    // Distance from obstacle radius where proximity penalty begins
    scalar_t temperature_lambda;  // Temperature parameter for MPPI weights

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Calculates a penalty based on the distance to the nearest obstacle.
 * 
 * @param state         Current robot state
 * @param obstacle_map  The map of spherical obstacles
 * @param cost_params   The cost weights and parameters for collisions
 * @return scalar_t     The computed collision penalty
 */
__host__ __device__ inline scalar_t compute_collision_cost(const State& state, const ObstacleMap& obstacle_map, const CostParams& cost_params) {
    scalar_t collision_cost = 0.0f;
    int num_obs = obstacle_map.num_obstacles;
    
    // We loop up to MAX_OBSTACLES with a conditional to ensure the compiler can unroll
    #pragma unroll
    for (int i = 0; i < ObstacleMap::MAX_OBSTACLES; ++i) {
        if (i < num_obs) {
            const Obstacle& obs = obstacle_map.obstacles[i];
            
            // Calculate Euclidean distance from robot to obstacle center
            scalar_t dist = (state.position - obs.position).norm();
            
            if (dist < obs.radius) {
                // Hard Penalty: Robot is inside the obstacle
                collision_cost += cost_params.weight_obstacle_collision;
            } else if (dist < (obs.radius + cost_params.obstacle_buffer_radius)) {
                // Soft Penalty: Robot is in the proximity buffer
                // Cost scales quadratically as the robot gets closer to the obstacle boundary
                scalar_t penetration = (obs.radius + cost_params.obstacle_buffer_radius) - dist;
                collision_cost += cost_params.weight_obstacle_proximity * (penetration * penetration);
            }
        }
    }
    
    return collision_cost;
}

/**
 * @brief Computes the scalar cost for a single timestep.
 * 
 * Designed to be called from within a __global__ CUDA kernel.
 * 
 * @param state        Current state of the SRBM quadruped.
 * @param control      Control inputs applied (12 GRFs).
 * @param target_state The target state to track.
 * @param obstacle_map The environment obstacle map.
 * @param cost_params  The weights for the cost function components.
 * @return scalar_t    The computed cost value for this step.
 */
__host__ __device__ inline scalar_t compute_step_cost(
    const State& state,
    const Control& control,
    const State& target_state,
    const ObstacleMap& obstacle_map,
    const CostParams& cost_params)
{
    scalar_t cost = 0.0f;

    // 1. Position error
    Eigen::Matrix<scalar_t, 3, 1> pos_err = state.position - target_state.position;
    scalar_t dist_sq = pos_err.squaredNorm();
    cost += cost_params.weight_position * dist_sq;

    // 2. Velocity damping & Parking Brake
    Eigen::Matrix<scalar_t, 3, 1> vel_err = state.linear_velocity - target_state.linear_velocity;
    if (dist_sq < 0.1f) {
        // Parking mode: within ~0.3m, apply strong braking to stop
        cost += 50.0f * cost_params.weight_velocity * vel_err.squaredNorm();
    } else {
        // Transit mode: milder distance-dependent braking so it doesn't freeze early
        scalar_t braking_factor = 1.0f + 5.0f / (1.0f + dist_sq);
        cost += cost_params.weight_velocity * braking_factor * vel_err.squaredNorm();
    }

    // 3. Orientation Error (keeping the body level, ignoring yaw)
    Eigen::Matrix<scalar_t, 3, 1> body_up(0.0f, 0.0f, 1.0f);
    Eigen::Matrix<scalar_t, 3, 1> world_up = state.orientation * body_up;
    scalar_t tilt = 1.0f - world_up.z();
    cost += cost_params.weight_orientation * tilt;

    // 4. Angular velocity damping
    Eigen::Matrix<scalar_t, 3, 1> ang_vel_err = state.angular_velocity - target_state.angular_velocity;
    cost += cost_params.weight_angular_velocity * ang_vel_err.squaredNorm();

    // 5. Obstacle Collision Penalty
    cost += compute_collision_cost(state, obstacle_map, cost_params);

    // 6. Control Penalty
    cost += cost_params.weight_control * control.joint_torques.squaredNorm();

    return cost;
}

/**
 * @brief Terminal cost: heavily penalizes the final-state position and velocity error.
 */
__host__ __device__ inline scalar_t compute_terminal_cost(
    const State& state,
    const State& target_state,
    const ObstacleMap& obstacle_map,
    const CostParams& cost_params)
{
    scalar_t cost = 0.0f;

    // Heavy terminal position penalty (10x)
    Eigen::Matrix<scalar_t, 3, 1> pos_err = state.position - target_state.position;
    scalar_t dist_sq = pos_err.squaredNorm();
    cost += 10.0f * cost_params.weight_position * dist_sq;

    // Terminal velocity penalty
    Eigen::Matrix<scalar_t, 3, 1> vel_err = state.linear_velocity - target_state.linear_velocity;
    if (dist_sq < 0.1f) {
        // Parking mode: must be perfectly stopped at end of horizon
        cost += 100.0f * cost_params.weight_velocity * vel_err.squaredNorm();
    } else {
        // Transit mode
        cost += 10.0f * cost_params.weight_velocity * vel_err.squaredNorm();
    }

    // Orientation at end should be level (ignoring yaw)
    Eigen::Matrix<scalar_t, 3, 1> body_up(0.0f, 0.0f, 1.0f);
    Eigen::Matrix<scalar_t, 3, 1> world_up = state.orientation * body_up;
    scalar_t tilt = 1.0f - world_up.z();
    cost += cost_params.weight_orientation * tilt;

    // Final obstacle check
    cost += compute_collision_cost(state, obstacle_map, cost_params);

    return cost;
}

} // namespace mppi

#endif // MPPI_COST_FUNCTION_H
