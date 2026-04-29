#include "../include/robot_types.h"
#include "../include/cost_function.h"
#include <curand_kernel.h>

namespace mppi {

// Structure to pass physical parameters to the GPU kernel.
// 16-byte aligned for efficient memory access.
struct alignas(16) RobotParams {
    scalar_t mass;
    Eigen::Matrix<scalar_t, 3, 3> inertia_body;
    Eigen::Matrix<scalar_t, 3, 3> inertia_inv_body;
    Eigen::Matrix<scalar_t, 3, 4> foot_pos_body; // Foot positions relative to CoM in body frame
    Eigen::Matrix<scalar_t, 3, 1> gravity;       // Gravity vector in world frame (e.g., [0, 0, -9.81])
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/**
 * @brief Computes the next state using Single Rigid Body Model (SRBM) dynamics with Euler integration.
 * 
 * Designed to be called from a __global__ kernel. Uses Eigen device-compatible functions.
 * Avoids branch divergence and uses coalesced structures.
 * 
 * @param state Current robot state (position, orientation, velocity, angular velocity)
 * @param control Control inputs (Assumes 12 Ground Reaction Forces: 4 legs x 3 forces in world frame)
 * @param params Physical robot parameters (mass, inertia, foot positions)
 * @param dt Time step for integration
 * @return State The next state
 */
__host__ __device__ State compute_dynamics(const State& state, const Control& control, const RobotParams& params, scalar_t dt) {
    State next_state = state;

    // Ground contact parameters
    // Standing height is the negative of the foot Z offset (feet are at -0.3 in body frame, so CoM is at +0.3)
    const scalar_t GROUND_HEIGHT = 0.3f;
    const scalar_t GROUND_CONTACT_THRESHOLD = 0.05f; // Allow small margin above standing height

    // 1. Extract current state
    Eigen::Matrix<scalar_t, 3, 1> p = state.position;
    Eigen::Quaternion<scalar_t> q = state.orientation;
    Eigen::Matrix<scalar_t, 3, 1> v = state.linear_velocity;
    Eigen::Matrix<scalar_t, 3, 1> w = state.angular_velocity; // in body frame

    // 2. Extract GRFs 
    // We repurpose the 'joint_torques' array from the Control struct to hold the 3D GRFs for 4 legs in world frame.
    Eigen::Matrix<scalar_t, 3, 1> f1_world = control.joint_torques.segment<3>(0);
    Eigen::Matrix<scalar_t, 3, 1> f2_world = control.joint_torques.segment<3>(3);
    Eigen::Matrix<scalar_t, 3, 1> f3_world = control.joint_torques.segment<3>(6);
    Eigen::Matrix<scalar_t, 3, 1> f4_world = control.joint_torques.segment<3>(9);

    // 3. Ground contact constraint: Check each foot individually
    const scalar_t CONTACT_RAMP = 0.05f; // Sharp ramp for feet (5cm)
    
    // Function to compute contact factor for a foot at world Z
    auto get_contact = [&](scalar_t foot_z) {
        if (foot_z > CONTACT_RAMP) return 0.0f;
        if (foot_z <= 0.0f) return 1.0f;
        scalar_t t = foot_z / CONTACT_RAMP;
        return 1.0f - t * t * (3.0f - 2.0f * t); // Smooth ramp
    };

    // Foot positions in world frame: p_foot = p_com + q * r_body
    Eigen::Matrix<scalar_t, 3, 1> r1_world = q * params.foot_pos_body.col(0);
    Eigen::Matrix<scalar_t, 3, 1> r2_world = q * params.foot_pos_body.col(1);
    Eigen::Matrix<scalar_t, 3, 1> r3_world = q * params.foot_pos_body.col(2);
    Eigen::Matrix<scalar_t, 3, 1> r4_world = q * params.foot_pos_body.col(3);

    scalar_t c1 = get_contact(p.z() + r1_world.z());
    scalar_t c2 = get_contact(p.z() + r2_world.z());
    scalar_t c3 = get_contact(p.z() + r3_world.z());
    scalar_t c4 = get_contact(p.z() + r4_world.z());

    // Clamp Z forces to non-negative and apply per-foot contact factor
    f1_world.z() = fmaxf(f1_world.z(), 0.0f) * c1;
    f1_world.x() *= c1; f1_world.y() *= c1;

    f2_world.z() = fmaxf(f2_world.z(), 0.0f) * c2;
    f2_world.x() *= c2; f2_world.y() *= c2;

    f3_world.z() = fmaxf(f3_world.z(), 0.0f) * c3;
    f3_world.x() *= c3; f3_world.y() *= c3;

    f4_world.z() = fmaxf(f4_world.z(), 0.0f) * c4;
    f4_world.x() *= c4; f4_world.y() *= c4;

    // Sum of forces in world frame
    Eigen::Matrix<scalar_t, 3, 1> f_total_world = f1_world + f2_world + f3_world + f4_world + (params.mass * params.gravity);

    // 4. Transform forces to body frame for torque calculation
    Eigen::Quaternion<scalar_t> q_inv = q.conjugate();
    
    Eigen::Matrix<scalar_t, 3, 1> f1_body = q_inv * f1_world;
    Eigen::Matrix<scalar_t, 3, 1> f2_body = q_inv * f2_world;
    Eigen::Matrix<scalar_t, 3, 1> f3_body = q_inv * f3_world;
    Eigen::Matrix<scalar_t, 3, 1> f4_body = q_inv * f4_world;

    // Sum of torques in body frame
    // Cross products of foot positions (r) with forces (f)
    Eigen::Matrix<scalar_t, 3, 1> r1 = params.foot_pos_body.col(0);
    Eigen::Matrix<scalar_t, 3, 1> r2 = params.foot_pos_body.col(1);
    Eigen::Matrix<scalar_t, 3, 1> r3 = params.foot_pos_body.col(2);
    Eigen::Matrix<scalar_t, 3, 1> r4 = params.foot_pos_body.col(3);

    Eigen::Matrix<scalar_t, 3, 1> tau_total_body = 
        r1.cross(f1_body) + r2.cross(f2_body) + r3.cross(f3_body) + r4.cross(f4_body);

    // 5. Compute accelerations
    // Linear acceleration (world frame)
    Eigen::Matrix<scalar_t, 3, 1> a = f_total_world / params.mass;
    
    // Angular acceleration (body frame): I * w_dot + w x (I * w) = tau
    // => w_dot = I_inv * (tau - w x (I * w))
    // We explicitly evaluate the intermediate expression to avoid Eigen lazy evaluation issues on CUDA
    Eigen::Matrix<scalar_t, 3, 1> iw = params.inertia_body * w;
    Eigen::Matrix<scalar_t, 3, 1> net_torque = tau_total_body - w.cross(iw);
    Eigen::Matrix<scalar_t, 3, 1> w_dot = params.inertia_inv_body * net_torque;

    // 6. Euler Integration
    // Position and Linear Velocity
    next_state.position = p + v * dt;
    next_state.linear_velocity = v + a * dt;

    // Angular velocity
    next_state.angular_velocity = w + w_dot * dt;

    // Orientation integration: q_next = q + 0.5 * q * w_body * dt
    Eigen::Quaternion<scalar_t> q_dot;
    q_dot.w() = -0.5f * (q.x() * w.x() + q.y() * w.y() + q.z() * w.z());
    q_dot.x() =  0.5f * (q.w() * w.x() + q.y() * w.z() - q.z() * w.y());
    q_dot.y() =  0.5f * (q.w() * w.y() - q.x() * w.z() + q.z() * w.x());
    q_dot.z() =  0.5f * (q.w() * w.z() + q.x() * w.y() - q.y() * w.x());

    next_state.orientation.w() = q.w() + q_dot.w() * dt;
    next_state.orientation.x() = q.x() + q_dot.x() * dt;
    next_state.orientation.y() = q.y() + q_dot.y() * dt;
    next_state.orientation.z() = q.z() + q_dot.z() * dt;
    
    // Normalize quaternion to prevent drift
    next_state.orientation.normalize(); 

    // 7. Ground contact constraint: enforce floor
    if (next_state.position.z() < GROUND_HEIGHT) {
        next_state.position.z() = GROUND_HEIGHT;
        if (next_state.linear_velocity.z() < 0.0f) {
            next_state.linear_velocity.z() = 0.0f; // Kill downward velocity
        }
    }

    return next_state;
}

/**
 * @brief MPPI Rollout Kernel for the SRBM Quadruped
 * 
 * Each CUDA block represents one Trajectory (Rollout).
 * Thread 0 simulates the trajectory over the horizon.
 * 
 * @param initial_state       Pointer to the starting state
 * @param base_controls       Pointer to the sequence of nominal controls (size: horizon)
 * @param params              Pointer to the robot physical parameters
 * @param target_state        Pointer to the goal state for cost evaluation
 * @param obstacle_map        Pointer to the environment obstacles
 * @param cost_params         Pointer to the cost weights
 * @param rng_states          Pointer to the array of curandState (size: num_rollouts)
 * @param d_trajectory_costs  Pointer to the output costs array (size: num_rollouts)
 * @param d_first_step_noise  Pointer to the output array of first-step noise (size: num_rollouts)
 * @param horizon             Number of timesteps (N)
 * @param dt                  Time step
 * @param noise_std           Standard deviation of the Gaussian noise
 */
__global__ void mppi_rollout_kernel(
    const State* initial_state,
    const Control* base_controls,
    const RobotParams* params,
    const State* target_state,
    const ObstacleMap* obstacle_map,
    const CostParams* cost_params,
    curandState* rng_states,
    scalar_t* d_trajectory_costs,
    Control* d_first_step_noise,
    scalar_t* d_debug_trajectories,
    int horizon,
    scalar_t dt,
    scalar_t noise_std
) {
    int rollout_idx = blockIdx.x;

    // Use shared memory for caching commonly accessed variables
    __shared__ State s_state;
    __shared__ RobotParams s_params;
    __shared__ State s_target;
    __shared__ ObstacleMap s_obstacle_map;
    __shared__ CostParams s_cost_params;

    if (threadIdx.x == 0) {
        s_state = *initial_state;
        s_params = *params;
        s_target = *target_state;
        s_obstacle_map = *obstacle_map;
        s_cost_params = *cost_params;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        curandState local_rng = rng_states[rollout_idx];
        scalar_t total_cost = 0.0f;

        for (int t = 0; t < horizon; ++t) {
            Control u = base_controls[t];
            Control noise;

            // Generate noise and add to controls for all 12 GRFs
            for (int i = 0; i < 12; ++i) {
                scalar_t n = noise_std * curand_normal(&local_rng);
                noise.joint_torques(i) = n;
                u.joint_torques(i) += n;
            }

            // Save the noise from the very first timestep
            if (t == 0) {
                d_first_step_noise[rollout_idx] = noise;
            }

            // Step the dynamics
            s_state = compute_dynamics(s_state, u, s_params, dt);

            // Save state for debugging if requested (every 100th rollout)
            if (d_debug_trajectories && blockIdx.x % 100 == 0) {
                int debug_rollout_idx = blockIdx.x / 100;
                // Assuming d_debug_trajectories has size [100 * horizon * 13]
                int debug_idx = (debug_rollout_idx * horizon * 13) + (t * 13);
                d_debug_trajectories[debug_idx + 0] = s_state.position.x();
                d_debug_trajectories[debug_idx + 1] = s_state.position.y();
                d_debug_trajectories[debug_idx + 2] = s_state.position.z();
                d_debug_trajectories[debug_idx + 3] = s_state.orientation.w();
                d_debug_trajectories[debug_idx + 4] = s_state.orientation.x();
                d_debug_trajectories[debug_idx + 5] = s_state.orientation.y();
                d_debug_trajectories[debug_idx + 6] = s_state.orientation.z();
                d_debug_trajectories[debug_idx + 7] = s_state.linear_velocity.x();
                d_debug_trajectories[debug_idx + 8] = s_state.linear_velocity.y();
                d_debug_trajectories[debug_idx + 9] = s_state.linear_velocity.z();
                d_debug_trajectories[debug_idx + 10] = s_state.angular_velocity.x();
                d_debug_trajectories[debug_idx + 11] = s_state.angular_velocity.y();
                d_debug_trajectories[debug_idx + 12] = s_state.angular_velocity.z();
            }

            // Compute running cost
            total_cost += compute_step_cost(s_state, u, s_target, s_obstacle_map, s_cost_params);
        }

        // Add terminal cost (heavily penalizes final position error)
        total_cost += compute_terminal_cost(s_state, s_target, s_obstacle_map, s_cost_params);

        // Save RNG and total cost
        rng_states[rollout_idx] = local_rng;
        d_trajectory_costs[rollout_idx] = total_cost;
    }
}

/**
 * @brief MPPI Update Kernel
 * 
 * Performs parallel reduction to compute the minimum cost and the sum of the weights,
 * and then computes the final weighted average of the first-step control noise.
 * 
 * IMPORTANT: This kernel must be launched with 1 block, and blockDim.x must be a 
 * power of 2 (e.g., 512 or 1024) that is greater than or equal to num_rollouts.
 * The shared memory size must be allocated dynamically (e.g., blockDim.x * sizeof(scalar_t)).
 * 
 * @param d_trajectory_costs  Array of total costs for each rollout
 * @param d_first_step_noise  Array of first-step noise for each rollout
 * @param base_controls       Pointer to the nominal control sequence (size: horizon)
 * @param cost_params         Pointer to the cost weights (containing temperature_lambda)
 * @param num_rollouts        Total number of rollouts
 * @param horizon             Number of timesteps in the control sequence
 */
__global__ void mppi_update_kernel(
    const scalar_t* d_trajectory_costs,
    const Control* d_first_step_noise,
    Control* base_controls,
    const CostParams* cost_params,
    int num_rollouts,
    int horizon
) {
    int tid = threadIdx.x;
    
    // Dynamically allocated shared memory for parallel reductions
    extern __shared__ scalar_t sdata[];
    
    // 1. Find minimum cost (rho) across all trajectories
    // Initialize out-of-bounds threads with a very large number for min-reduction
    scalar_t my_cost = (tid < num_rollouts) ? d_trajectory_costs[tid] : 1e30f;
    sdata[tid] = my_cost;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }
    
    scalar_t rho = sdata[0];
    __syncthreads();
    
    // 2. Compute weights: w_i = exp(- (S_i - rho) / lambda) and sum them
    scalar_t lambda = cost_params->temperature_lambda;
    if (lambda < 1e-6f) lambda = 1e-6f; // Prevent division by zero
    
    // Initialize out-of-bounds threads with 0.0f for sum-reduction
    scalar_t my_weight = 0.0f;
    if (tid < num_rollouts) {
        my_weight = expf(-(my_cost - rho) / lambda);
    }
    
    sdata[tid] = my_weight;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    scalar_t sum_w = sdata[0];
    __syncthreads();
    
    // 3. Compute the final control update (Delta u) for the first timestep
    // We compute the weighted sum of the noise for each of the 12 GRFs.
    if (sum_w > 1e-6f) {
        for (int d = 0; d < 12; ++d) {
            scalar_t my_weighted_noise = 0.0f;
            if (tid < num_rollouts) {
                my_weighted_noise = my_weight * d_first_step_noise[tid].joint_torques(d);
            }
            sdata[tid] = my_weighted_noise;
            __syncthreads();
            
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                __syncthreads();
            }
            
            // Thread 0 applies the update to the base_controls
            if (tid == 0) {
                scalar_t delta_u = sdata[0] / sum_w;
                base_controls[0].joint_torques(d) += delta_u;
            }
            __syncthreads();
        }
    }
}

} // namespace mppi
