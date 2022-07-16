cmake_minimum_required(VERSION 2.8.12)
project(ceres_python)

set(PYBIND11_CPP_STANDARD -std=c++14)
add_subdirectory(ext/ceres_pybind/pybind11)
pybind11_add_module(ceres_python cpp_src/bindings.cpp)
target_link_libraries(ceres_python PRIVATE ceres)

find_package(Eigen3 REQUIRED)
include_directories("${EIGEN_INCLUDE_DIR}")

# # Build shared library for ceres-solver
set(BUILD_SHARED_LIBS ON)
add_subdirectory(ext/ceres_pybind/ceres-solver)
include_directories(ext/ceres_pybind/ceres-solver/include)
