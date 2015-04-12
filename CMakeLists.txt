
# ==================================================================================================
# This file is part of the CLTune project.
#
# Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
#
# -------------------------------------------------------------------------------------------------
#
# Copyright 2014 SURFsara
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==================================================================================================

# CMake project
cmake_minimum_required(VERSION 2.8)
project("cltune" C CXX)
set(cltune_VERSION_MAJOR 1)
set(cltune_VERSION_MINOR 3)
set(cltune_VERSION_PATCH 2)

# Options
option(SAMPLES "Enable compilation of sample programs" ON)
option(TESTS "Enable compilation of the Google tests" OFF)

# ==================================================================================================

# Compiler-version check
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
        message(FATAL_ERROR "GCC version must be at least 4.9 (for full C++11 compatibility)")
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
    ${OPENCL_INCLUDE_DIRS})

# Gathers all source-files
set(TUNER
    src/tuner.cc
    src/kernel_info.cc
    src/string_range.cc
    src/memory.cc
    src/opencl.cc
    src/searcher.cc
    src/searchers/full_search.cc
    src/searchers/random_search.cc
    src/searchers/annealing.cc)

# Creates and links the library
add_library(cltune SHARED ${TUNER})
target_link_libraries(cltune ${OPENCL_LIBRARIES})

# Installs the library
install(TARGETS cltune DESTINATION lib)
install(FILES include/cltune.h include/cl.hpp DESTINATION include)

# ==================================================================================================
# Optional: Enables compilation of sample programs
if (SAMPLES)

    # Adds sample programs
    add_executable(sample_simple samples/simple.cc)
    add_executable(sample_gemm samples/gemm.cc)
    add_executable(sample_gemm_annealing samples/gemm_annealing.cc)
    target_link_libraries(sample_simple cltune ${OPENCL_LIBRARIES} ${OpenMP_LIBRARY})
    target_link_libraries(sample_gemm cltune ${OPENCL_LIBRARIES} ${OpenMP_LIBRARY})
    target_link_libraries(sample_gemm_annealing cltune ${OPENCL_LIBRARIES} ${OpenMP_LIBRARY})
    install(TARGETS sample_simple DESTINATION bin)
    install(TARGETS sample_gemm DESTINATION bin)
    install(TARGETS sample_gemm_annealing DESTINATION bin)

endif()
# ==================================================================================================
# Optional: Enables compilation of the Google tests
if (TESTS)

    # Enables Google Test tests (source-code is shipped with the project)
    add_subdirectory(external/gtest-1.7.0)
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

    # Compiles the tests
    add_executable(unit_tests test/tuner.cc test/kernel_info.cc)
    target_link_libraries(unit_tests gtest gtest_main cltune ${OPENCL_LIBRARIES})
    install(TARGETS unit_tests DESTINATION bin)

    # Adds the tests
    add_test(
        name unit_tests
        command unit_tests
    )

endif()
# ==================================================================================================
