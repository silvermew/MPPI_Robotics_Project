cmake_minimum_required(VERSION 3.24)
project(MPPI_Robotics_Project LANGUAGES CXX CUDA)

# Target modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA for NVIDIA Blackwell (sm_90) / Ada / Hopper architectures, falling back to native for local testing
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Relaxed constexpr is critical for Eigen to work inside __device__ kernels
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# We use FetchContent to automatically download and link Eigen and GoogleTest
include(FetchContent)

# Disable testing for third-party libraries so CTest doesn't run their test suites
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for third-party libraries" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)

# Fetch Eigen 3.4.0 (Header-only library)
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_MakeAvailable(Eigen3)

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows MSVC: Prevent GTest from overriding the parent project's CRT settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --- Define the Test Executable ---
enable_testing()

add_executable(test_dynamics tests/test_dynamics.cu)

# Include the root directory so we can resolve includes like "include/robot_types.h"
target_include_directories(test_dynamics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link against Eigen and GTest
target_link_libraries(test_dynamics
    PRIVATE
    Eigen3::Eigen
    GTest::gtest_main
)

# Automatically discover and register GTest cases for CTest
include(GoogleTest)
gtest_discover_tests(test_dynamics)

# --- MPPI Integration Tests ---
add_executable(test_mppi_integration tests/test_mppi_integration.cu)
target_include_directories(test_mppi_integration PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_mppi_integration
    PRIVATE
    Eigen3::Eigen
    GTest::gtest_main
)
gtest_discover_tests(test_mppi_integration)

# --- Full Loop Convergence Tests ---
add_executable(test_convergence tests/test_convergence.cu)
target_include_directories(test_convergence PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_convergence
    PRIVATE
    Eigen3::Eigen
    GTest::gtest_main
)
gtest_discover_tests(test_convergence)

# --- Obstacle Avoidance Tests ---
add_executable(test_obstacle_avoidance tests/test_obstacle_avoidance.cu)
target_include_directories(test_obstacle_avoidance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_obstacle_avoidance
    PRIVATE
    Eigen3::Eigen
    GTest::gtest_main
)
gtest_discover_tests(test_obstacle_avoidance)

# --- Main Simulation Executable ---
add_executable(mppi_sim src/main.cu)
target_include_directories(mppi_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mppi_sim
    PRIVATE
    Eigen3::Eigen
)

