# ==================================================================================================
# This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
# a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
#
# Defines the following variables:
#   OPENCL_FOUND          Boolean holding whether or not the OpenCL library was found
#   OPENCL_INCLUDE_DIRS   The OpenCL include directory
#   OPENCL_LIBRARIES      The OpenCL library
#
# In case OpenCL is not installed in the default directory, set the OPENCL_ROOT variable to point to
# the root of OpenCL, such that 'OpenCL/cl.h' or 'CL/cl.h' can be found in $OPENCL_ROOT/include.
# This can either be done using an environmental variable (e.g. export OPENCL_ROOT=/path/to/opencl)
# or using a CMake variable (e.g. cmake -DOPENCL_ROOT=/path/to/opencl ..).
#
# --------------------------------------------------------------------------------------------------
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

# Sets the possible install locations
set(OPENCL_HINTS
  ${OPENCL_ROOT}
  $ENV{OPENCL_ROOT}
  $ENV{AMDAPPSDKROOT}
  $ENV{CUDA_PATH}
  $ENV{INTELOCLSDKROOT}
  $ENV{NVSDKCOMPUTE_ROOT}
  $ENV{ATISTREAMSDKROOT}
)
set(OPENCL_PATHS
  /usr/local/cuda
  /opt/cuda
  /opt/intel/opencl
  /usr
  /usr/local
)

# Finds the include directories
find_path(OPENCL_INCLUDE_DIRS
  NAMES OpenCL/cl.h CL/cl.h
  HINTS ${OPENCL_HINTS}
  PATH_SUFFIXES include OpenCL/common/inc inc include/x86_64 include/x64
  PATHS ${OPENCL_PATHS}
  DOC "OpenCL include header OpenCL/cl.h or CL/cl.h"
)
mark_as_advanced(OPENCL_INCLUDE_DIRS)

# Finds the library
find_library(OPENCL_LIBRARIES
  NAMES OpenCL
  HINTS ${OPENCL_HINTS}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x86_64/sdk lib/x64 lib/x86 lib/Win32 OpenCL/common/lib/x64
  PATHS ${OPENCL_PATHS}
  DOC "OpenCL library"
)
mark_as_advanced(OPENCL_LIBRARIES)

# ==================================================================================================

# Notification messages
if(NOT OPENCL_INCLUDE_DIRS)
    message(STATUS "Could NOT find 'OpenCL/cl.h' or 'CL/cl.h', install OpenCL or set OPENCL_ROOT")
endif()
if(NOT OPENCL_LIBRARIES)
    message(STATUS "Could NOT find OpenCL library, install it or set OPENCL_ROOT")
endif()

# Determines whether or not OpenCL was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL DEFAULT_MSG OPENCL_INCLUDE_DIRS OPENCL_LIBRARIES)

# ==================================================================================================
