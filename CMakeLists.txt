cmake_minimum_required(VERSION 3.1)
project(VoxelFEM)

if (${CMAKE_VERSION} VERSION_GREATER "3.13")
    cmake_policy(SET CMP0076 NEW)
endif()

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile flags (used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Make sure warnings/errors are still colorized when using Ninja for building.
add_definitions(-fdiagnostics-color=always)

# For python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#Include MeshFem (Eigen will be included by MeshFEM)
if (NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    if (CMAKE_BUILD_TYPE MATCHES DEBUG)
        option(MESHFEM_VECTORIZE "" OFF)
    else()
        option(MESHFEM_VECTORIZE "" ON) # Request vectorization
    endif()
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

option(VOXELFEM_FINE_BENCHMARK "Whether to report benchmarks for the sub-components of the PCG solver." ON)
option(VOXELFEM_DISABLE_3D     "Whether to disable the disable the 3D bindings to accelerate build times" OFF)
set(VOXELFEM_SIMD_WIDTH  "4"  CACHE STRING  "Width to use for manual vectorization of applyK, etc.")

# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
# set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

add_library(VoxelFEM SHARED
    TensorProductBasisPolynomial.hh
    TensorProductPolynomialInterpolant.hh
    TensorProductQuadrature.hh
    TensorProductQuadrature.cc
    TensorProductSimulator.hh
)

target_link_libraries(VoxelFEM PUBLIC MeshFEM)

if (VOXELFEM_FINE_BENCHMARK)
    target_compile_definitions(VoxelFEM PUBLIC -DVOXELFEM_FINE_BENCHMARK)
endif()
if (VOXELFEM_DISABLE_3D)
    target_compile_definitions(VoxelFEM PUBLIC -DVOXELFEM_DISABLE_3D)
endif()

target_compile_definitions(VoxelFEM PUBLIC -DVOXELFEM_SIMD_WIDTH=${VOXELFEM_SIMD_WIDTH})

add_subdirectory(python_bindings)
