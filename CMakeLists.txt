
# ==================================================================================================
# This file is part of the CLTune project. The project is licensed under the MIT license by
# SURFsara, (c) 2014.
#
# Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
#
# ==================================================================================================

# CMake project
cmake_minimum_required(VERSION 2.8)
project("cltune" C CXX)

# Options
option(ENABLE_SAMPLES "Enable compilation of sample programs" ON)
option(ENABLE_TESTS "Enable compilation of the Google tests" OFF)

# ==================================================================================================

# Compiler-version check
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
        message(FATAL_ERROR "GCC version must be at least 4.7 (for C++11)")
    endif()
endif()

# C++11 compiler settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wall -Wno-comment")

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Requires OpenCL (FindOpenCL is included as part of this project)
find_package(OpenCL REQUIRED)

# ==================================================================================================

# The includes
include_directories(
    ${cltune_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIR})

# Gathers all source-files
set(TUNER
    src/tuner.cc
    src/kernel_info.cc
    src/string_range.cc
    src/memory.cc
    src/opencl.cc)

# Links the library
add_library(cltune SHARED ${TUNER})
target_link_libraries(cltune ${OpenCL_LIBRARY})

# ==================================================================================================
# Optional: Enable compilation of sample programs
if (ENABLE_SAMPLES)

    # Adds sample programs
    add_executable(sample_gemm samples/gemm.cc)
    add_executable(sample_simple samples/simple.cc)
    target_link_libraries(sample_gemm cltune ${OpenCL_LIBRARY} ${OpenMP_LIBRARY})
    target_link_libraries(sample_simple cltune ${OpenCL_LIBRARY} ${OpenMP_LIBRARY})

endif()
# ==================================================================================================
# Optional: Enable compilation of the Google tests
if (ENABLE_TESTS)

    # Enables Google Test tests (source-code is shipped with the project)
    add_subdirectory(external/gtest-1.7.0)
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

    # Compiles the tests
    add_executable(unit_tests test/tuner.cc test/kernel_info.cc)
    target_link_libraries(unit_tests gtest gtest_main cltune ${OpenCL_LIBRARY})

    # Adds the tests
    add_test(
        name unit_tests
        command unit_tests
    )

endif()
# ==================================================================================================
