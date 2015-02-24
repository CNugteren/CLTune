
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the main OpenCL class which is used to initialize and tear down an OpenCL
// environment, including a single OpenCL platform, device, context, and queue. In turn, this class
// relies on the C++ OpenCL header, which is a wrapper for the C OpenCL API.
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

#ifndef CLBLAS_TUNER_OPENCL_H_
#define CLBLAS_TUNER_OPENCL_H_

#include <string>
#include <vector>
#include <stdexcept>

// The C++ OpenCL wrapper
#include "cl.hpp"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class OpenCL {
 public:

  // Types of devices to consider
  const cl_device_type kDeviceType = CL_DEVICE_TYPE_ALL;

  // Initializes the OpenCL platform, device, and creates a context and a queue
  explicit OpenCL(const size_t platform_id, const size_t device_id);

  // Accessors
  cl::Device device() const { return device_; }
  cl::Context context() const { return context_; }
  cl::CommandQueue queue() const { return queue_; }

  // Checks whether the global and local thread-sizes, and local memory size are compatible with the
  // current device
  size_t VerifyThreadSizes(const cl::NDRange &global, const cl::NDRange &local) const;
  void VerifyLocalMemory(const size_t local_memory) const;
  
 private:

  // Converts an unsigned integer to a string by first casting it to a long long integer. This is
  // required for older compilers that do not fully implement std::to_string (part of C++11).
  std::string ToString(int value) const { return std::to_string(static_cast<long long>(value)); }

  // Settings
  bool suppress_output_;

  // OpenCL variables
  cl::Platform platform_;
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;

  // OpenCL device properties and limitations
  size_t max_local_dims_;
  size_t max_local_threads_;
  std::vector<size_t> max_local_sizes_;
  size_t local_memory_size_;
};

// =================================================================================================
} // namespace cltune

// CLBLAS_TUNER_OPENCL_H_
#endif
