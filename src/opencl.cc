
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the OpenCL class (see the header for information about the class).
//
// -------------------------------------------------------------------------------------------------
//
// Copyright 2014 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

// The corresponding header file
#include "internal/opencl.h"

namespace cltune {
// =================================================================================================

// Messages printed to stdout (in colours)
const std::string OpenCL::kMessageFull = "\x1b[32m[==========]\x1b[0m";

// =================================================================================================

// Gets a list of all platforms/devices and chooses the selected ones. Initializes OpenCL and also
// downloads properties of the device for later use.
OpenCL::OpenCL(const size_t platform_id, const size_t device_id):
    suppress_output_{false} {

  // Starting on a new platform/device
  if (!suppress_output_) {
    fprintf(stdout, "\n%s Initializing OpenCL on platform %lu device %lu\n",
            kMessageFull.c_str(), platform_id, device_id);
  }

  // Initializes the OpenCL platform
  auto platforms = std::vector<cl::Platform>{};
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }
  if (platform_id >= platforms.size()) {
    throw std::runtime_error("Invalid OpenCL platform number: " + ToString(platform_id));
  }
  platform_ = platforms[platform_id];

  // Initializes the OpenCL device
  auto devices = std::vector<cl::Device>{};
  platform_.getDevices(kDeviceType, &devices);
  if (devices.size() == 0) {
    throw std::runtime_error("No OpenCL devices found on platform " + ToString(platform_id));
  }
  if (device_id >= devices.size()) {
    throw std::runtime_error("Invalid OpenCL device number: " + ToString(device_id));
  }
  device_ = devices[device_id];

  // Creates the context and the queue
  //context_ = cl::Context({device_});
  auto status = CL_SUCCESS;
  context_ = clCreateContext(nullptr, 1, &(device_()), nullptr, nullptr, &status);
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Context creation error", status); }
  queue_ = clCreateCommandQueue(context_, device_(), CL_QUEUE_PROFILING_ENABLE, &status);
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Command queue creation error", status); }

  // Gets platform and device properties
  auto opencl_version = device_.getInfo<CL_DEVICE_VERSION>();
  device_name_        = device_.getInfo<CL_DEVICE_NAME>();
  max_local_dims_     = device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
  max_local_threads_  = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  max_local_sizes_    = device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  local_memory_size_  = device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

  // Prints the device name
  if (!suppress_output_) {
    fprintf(stdout, "%s Device name: '%s' (%s)\n", kMessageFull.c_str(),
            device_name_.c_str(), opencl_version.c_str());
  }
}

// Releases the OpenCL objects
OpenCL::~OpenCL() {
  clReleaseCommandQueue(queue_);
  clReleaseContext(context_);
}

// =================================================================================================

// Verifies: 1) the local worksize in each dimension, 2) the local worksize in all dimensions
// combined, and 3) the number of dimensions. For now, the global size is not verified.
bool OpenCL::ValidThreadSizes(const IntRange &global, const IntRange &local) const {
  auto local_size = size_t{1};
  auto global_size = size_t{1};
  for (auto &item: global) { global_size *= item; }
  for (auto &item: local) { local_size *= item; }
  for (auto i=size_t{0}; i<local.size(); ++i) {
    if (local[i] > max_local_sizes_[i]) { return false; }
  }
  if (local_size > max_local_threads_) { return false; }
  if (local.size() > max_local_dims_) { return false; }
  return true;
}

// Returns the total local size
size_t OpenCL::GetLocalSize(const IntRange &global, const IntRange &local) const {
  auto local_size = size_t{1};
  for (auto &item: local) { local_size *= item; }
  return local_size;
}

// Verifies the local memory usage of the kernel (provided as argument) against the device
// limitation (obtained in the constructor).
bool OpenCL::ValidLocalMemory(const size_t local_memory) const {
  if (local_memory > local_memory_size_) { return false; }
  return true;
}

// =================================================================================================
} // namespace cltune
