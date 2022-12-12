
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

cmake_minimum_required(VERSION 2.8.11)

# Overrides for MSVC static runtime
set(CMAKE_USER_MAKE_RULES_OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c_flag_overrides.cmake)
set(CMAKE_USER_MAKE_RULES_OVERRIDE_CXX ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cxx_flag_overrides.cmake)

# CMake project details
project("cltune" CXX)
set(cltune_VERSION_MAJOR 2)
set(cltune_VERSION_MINOR 7)
set(cltune_VERSION_PATCH 0)

set(cltune_VERSION "${cltune_VERSION_MAJOR}.${cltune_VERSION_MINOR}.${cltune_VERSION_PATCH}")
set(cltune_SOVERSION 2 )

# Options and their default values
option(BUILD_SHARED_LIBS "Build a shared (ON) or static library (OFF)" ON)
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

# Compile in verbose mode with additional diagnostic messages
option(VERBOSE "Compile in verbose mode for additional diagnostic messages" OFF)
if(VERBOSE)
  message("-- Building in verbose mode")
  add_definitions(-DVERBOSE)
endif()

# ==================================================================================================

# RPATH settings
set(CMAKE_MACOSX_RPATH 1)

# ==================================================================================================

# Compiler-version check (requires at least CMake 2.8.10)
if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
    message(FATAL_ERROR "GCC version must be at least 4.7")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.3)
    message(FATAL_ERROR "Clang version must be at least 3.3")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL AppleClang)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "AppleClang version must be at least 5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
    message(FATAL_ERROR "ICC version must be at least 14.0")
  endif()
elseif(MSVC)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
    message(FATAL_ERROR "MS Visual Studio version must be at least 18.0")
  endif()
endif()

# DLL Settings
if(MSVC)
  if(BUILD_SHARED_LIBS)
    add_definitions(" /DCLTUNE_DLL")
  endif()
endif(MSVC)

# C++ compiler settings
if(MSVC)
  set(FLAGS "/Ox")
  set(FLAGS "${FLAGS} /wd4715 /wd4996")
else()
  set(FLAGS "-O3 -std=c++11")
  # Silence warnings about not specifying ocl version
  set(FLAGS "${FLAGS} -DCL_TARGET_OPENCL_VERSION=120")
  if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(FLAGS "${FLAGS} -Wall -Wno-comment")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8.4)
      set(FLAGS "${FLAGS} -Wno-attributes")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    set(FLAGS "${FLAGS} -Wextra -Wno-c++98-compat -Wno-c++98-compat-pedantic")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS}")

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
if(BUILD_SHARED_LIBS)
  add_library(cltune SHARED ${TUNER})
  set_target_properties(cltune PROPERTIES VERSION ${cltune_VERSION})
  set_target_properties(cltune PROPERTIES SOVERSION ${cltune_SOVERSION})
else(BUILD_SHARED_LIBS)
  add_library(cltune STATIC ${TUNER})
endif()
target_link_libraries(cltune ${FRAMEWORK_LIBRARIES})

# Sets the proper __declspec(dllexport) keyword for Visual Studio when the library is built
if(MSVC)
  if(BUILD_SHARED_LIBS)
    target_compile_definitions(cltune PRIVATE COMPILING_DLL=1) # requires at least CMake 2.8.11
  endif()
endif()

if(UNIX)
    # Installs the library
    include("GNUInstallDirs")
    install(TARGETS cltune DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(FILES include/cltune.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    # Install pkg-config file on Linux
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cltune.pc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cltune.pc" @ONLY IMMEDIATE)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cltune.pc
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
else(UNIX)
     # Installs the library
     install(TARGETS cltune DESTINATION lib)
     install(FILES include/cltune.h DESTINATION include)
endif()

# ==================================================================================================

# Optional: Enables compilation of sample programs
if (SAMPLES)

  # Adds sample programs
  add_executable(sample_simple samples/simple/simple.cc)
  add_executable(sample_conv_simple samples/conv_simple/conv_simple.cc)
  add_executable(sample_multiple_kernels samples/multiple_kernels/multiple_kernels.cc)
  add_executable(sample_gemm samples/gemm/gemm.cc)
  add_executable(sample_conv samples/conv/conv.cc)
  target_link_libraries(sample_simple cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_conv_simple cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_multiple_kernels cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_gemm cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})
  target_link_libraries(sample_conv cltune ${FRAMEWORK_LIBRARIES} ${OpenMP_LIBRARY})

  # Note: these are not installed because they depend on their separate OpenCL kernel files
endif()

# ==================================================================================================

# Optional: Enable inclusion of the test-suite
if (TESTS)
  enable_testing()
  include_directories(${cltune_SOURCE_DIR}/test ${cltune_SOURCE_DIR}/include ${FRAMEWORK_INCLUDE_DIRS})
  add_executable(unit_tests
                 test/main.cc
                 test/clcudaapi.cc
                 test/tuner.cc
                 test/kernel_info.cc)
  target_link_libraries(unit_tests cltune ${FRAMEWORK_LIBRARIES})
  add_test(unit_tests unit_tests)
endif()

# ==================================================================================================
