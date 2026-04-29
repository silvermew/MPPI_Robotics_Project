#ifndef MPPI_ROBOT_TYPES_H
#define MPPI_ROBOT_TYPES_H

#include <Eigen/Dense>

namespace mppi {

// Define scalar type for simple precision swapping (float for GPU speed, double for precision)
using scalar_t = float;

/**
 * @brief State struct for a Single Rigid Body Model (SRBM) of a quadruped.
 * 
 * Represents the floating base state. 
 * Memory is 16-byte aligned to ensure efficient, coalesced memory access on GPU.
 */
struct alignas(16) State {
    Eigen::Matrix<scalar_t, 3, 1> position;         // 3D Position (x, y, z) in world frame
    Eigen::Quaternion<scalar_t>   orientation;      // Orientation (w, x, y, z)
    Eigen::Matrix<scalar_t, 3, 1> linear_velocity;  // Linear velocity (vx, vy, vz)
    Eigen::Matrix<scalar_t, 3, 1> angular_velocity; // Angular velocity (wx, wy, wz)

    // Required for proper memory alignment when using fixed-size Eigen members
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Control struct for a quadruped.
 * 
 * Represents the joint torques applied to the 12 DOFs of the legs.
 * Memory is 16-byte aligned to ensure efficient, coalesced memory access on GPU.
 */
struct alignas(16) Control {
    Eigen::Matrix<scalar_t, 12, 1> joint_torques;   // Torques for the 12 leg joints

    // Required for proper memory alignment when using fixed-size Eigen members
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Represents a spherical obstacle in the environment.
 */
struct alignas(16) Obstacle {
    Eigen::Matrix<scalar_t, 3, 1> position;
    scalar_t radius;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief ObstacleMap struct for storing multiple spherical obstacles.
 * Maximum of 10 obstacles for static memory allocation on GPU.
 */
struct alignas(16) ObstacleMap {
    static constexpr int MAX_OBSTACLES = 10;
    Obstacle obstacles[MAX_OBSTACLES];
    int num_obstacles;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace mppi

#endif // MPPI_ROBOT_TYPES_H
