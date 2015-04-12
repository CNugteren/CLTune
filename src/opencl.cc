
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

#include "tuner/internal/opencl.h"

// Include other classes
#include "tuner/tuner.h"

namespace cltune {
// =================================================================================================

// Gets a list of all platforms/devices and chooses the selected ones. Initializes OpenCL and also
// downloads properties of the device for later use.
OpenCL::OpenCL(const size_t platform_id, const size_t device_id):
    suppress_output_{false} {

  // Starting on a new platform/device
  if (!suppress_output_) {
    fprintf(stdout, "\n%s Initializing OpenCL on platform %lu device %lu\n",
            Tuner::kMessageFull.c_str(), platform_id, device_id);
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
  context_ = cl::Context({device_});
  queue_ = cl::CommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE);

  // Gets platform and device properties
  auto opencl_version = device_.getInfo<CL_DEVICE_VERSION>();
  device_name_        = device_.getInfo<CL_DEVICE_NAME>();
  max_local_dims_     = device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
  max_local_threads_  = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  max_local_sizes_    = device_.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  local_memory_size_  = device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

  // Prints the device name
  if (!suppress_output_) {
    fprintf(stdout, "%s Device name: '%s' (%s)\n", Tuner::kMessageFull.c_str(),
            device_name_.c_str(), opencl_version.c_str());
  }
}

// =================================================================================================

// Verifies: 1) the local worksize in each dimension, 2) the local worksize in all dimensions
// combined, and 3) the number of dimensions. For now, the global size is not verified.
size_t OpenCL::VerifyThreadSizes(const cl::NDRange &global, const cl::NDRange &local) const {
  auto local_size = 1UL;
  auto global_size = 1UL;
  for (auto i=0UL; i<global.dimensions(); ++i) { global_size *= global[i]; }
  for (auto i=0UL; i<local.dimensions(); ++i) {
    local_size *= local[i];
    if (local[i] > max_local_sizes_[i]) {
      throw std::runtime_error("Local size in dimension "+ToString(i)+
                               " larger than "+ToString(max_local_sizes_[i]));
    }
  }
  if (local_size > max_local_threads_) {
    throw std::runtime_error("Local size larger than "+ToString(max_local_threads_));
  }
  if (local.dimensions() > max_local_dims_) {
    throw std::runtime_error("More thread-dimensions than "+ToString(max_local_dims_));
  }
  return local_size;
}

// Verifies the local memory usage of the kernel (provided as argument) against the device
// limitation (obtained in the constructor).
void OpenCL::VerifyLocalMemory(const size_t local_memory) const {
  if (local_memory > local_memory_size_) {
    throw std::runtime_error("Local memory size larger than "+ToString(local_memory_size_));
  }
}

// =================================================================================================
} // namespace cltune
