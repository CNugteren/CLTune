
# ==================================================================================================
# This file is part of the CLTune project.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
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
cmake_minimum_required(VERSION 2.8.10)
project("cltune" CXX)
set(cltune_VERSION_MAJOR 1)
set(cltune_VERSION_MINOR 7)
set(cltune_VERSION_PATCH 1)

# Options
option(SAMPLES "Enable compilation of sample programs" ON)
option(TESTS "Enable compilation of the Google tests" OFF)

# Select between OpenCL and CUDA back-end
option(USE_OPENCL "Use OpenCL instead of CUDA" ON)
if(USE_OPENCL)
  message("-- Building with OpenCL")
  add_definitions(-DUSE_OPENCL)
else()
  message("-- Building with CUDA")
endif()

# ==================================================================================================

# RPATH settings
set(CMAKE_SKIP_BUILD_RPATH false) # Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_BUILD_WITH_INSTALL_RPATH false) # When building, don't use the install RPATH already
set(CMAKE_INSTALL_RPATH "") # The RPATH to be used when installing
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH false) # Don't add the automatically determined parts

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "Clang version must be at least 5.0")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# C++ compiler settings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(FLAGS "/Ox")
else ()
  set(FLAGS "-O3 -std=c++11")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(FLAGS "${FLAGS} -Wall -Wno-comment")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.4)
      set(FLAGS "${FLAGS} -Wno-attributes")
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(FLAGS "${FLAGS} -Wextra")
  endif()
endif()

# ==================================================================================================

# Package scripts location
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Requires CUDA or OpenCL. The latter is found through the included "FindOpenCL.cmake".
if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  set(FRAMEWORK_INCLUDE_DIRS ${OPENCL_INCLUDE_DIRS})
  set(FRAMEWORK_LIBRARY_DIRS )
  set(FRAMEWORK_LIBRARIES ${OPENCL_LIBRARIES})
else()
  find_package(CUDA REQUIRED)
  set(FRAMEWORK_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  set(FRAMEWORK_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  set(FRAMEWORK_LIBRARIES cuda nvrtc)
endif()

# ==================================================================================================

# Include directories: CLTune headers and OpenCL/CUDA includes
include_directories(${cltune_SOURCE_DIR}/include ${FRAMEWORK_INCLUDE_DIRS})

# Link directories: CUDA toolkit
link_directories(${FRAMEWORK_LIBRARY_DIRS})

# Gathers all source-files
set(TUNER
    src/cltune.cc
    src/tuner_impl.cc
    src/kernel_info.cc
    src/searcher.cc
    src/searchers/full_search.cc
    src/searchers/random_search.cc
    src/searchers/annealing.cc
    src/searchers/pso.cc
    src/ml_model.cc
    src/ml_models/linear_regression.cc
    src/ml_models/neural_network.cc)

# Creates and links the library
add_library(cltune SHARED ${TUNER})
target_link_libraries(cltune ${FRAMEWORK_LIBRARIES})

# Installs the library
install(TARGETS cltune DESTINATION lib)
install(FILES include/cltune.h DESTINATION include)

# ==================================================================================================
# Optional: Enables compilation of sample programs
if (SAMPLES)

  # Adds sample programs
  add_executable(sample_simple samples/simple/simple.cc)
  add_executable(sample_gemm samples/gemm/gemm.cc)
  add_executable(sample_conv samples/conv/conv.cc)
  target_link_libraries(sample_simple cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_gemm cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_conv cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})

  # Note: these are not installed because they depend on their separate OpenCL kernel files

endif()
# ==================================================================================================
# Optional: Enables compilation of the Google tests
if (TESTS)

  # The tests use specific flags to reduce the amount of warnings from GTest.
  set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

  # Enables Google Test tests (source-code is shipped with the project)
  add_subdirectory(external/gtest-1.7.0)
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  # Compiles the tests
  add_executable(unit_tests test/tuner.cc test/kernel_info.cc)
  target_link_libraries(unit_tests gtest gtest_main cltune ${FRAMEWORK_LIBRARIES})

  # Adds the tests
  add_test(name unit_tests command unit_tests)
endif()
# ==================================================================================================
