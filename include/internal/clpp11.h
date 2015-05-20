
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements a C++11 wrapper around some OpenCL C data-types, similar to Khronos' cl.hpp:
// https://www.khronos.org/registry/cl/api/1.1/cl.hpp
// The main differences are modern C++11 support and an implemenation of only the basic needs (for
// this project).
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

#ifndef CLTUNE_CLPP11_H_
#define CLTUNE_CLPP11_H_

#include <utility> // std::swap
#include <algorithm> // std::copy
#include <string> // std::string
#include <vector> // std::vector

// Includes the normal OpenCL C header
#if defined(__APPLE__) || defined(__MACOSX)
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

namespace cltune {
// =================================================================================================

// C++11 version of cl_program
class Program {
 public:

  // Memory management
  Program(const cl_context context, const std::string &source):
    length_(source.length()) {
      std::copy(source.begin(), source.end(), back_inserter(source_));
      source_ptr_ = source_.data();
      program_ = clCreateProgramWithSource(context, 1, &source_ptr_, &length_, nullptr);
    }
  ~Program() {
    clReleaseProgram(program_);
  }
  Program(const Program& other):
      length_(other.length_),
      source_(other.source_),
      source_ptr_(other.source_ptr_),
      program_(other.program_) {
    clRetainProgram(program_);
  }
  Program& operator=(Program other) {
    swap(*this, other);
    return *this;
  }
  void swap(Program& first, Program& second) {
    std::swap(first.length_, second.length_);
    std::swap(first.source_, second.source_);
    std::swap(first.source_ptr_, second.source_ptr_);
    std::swap(first.program_, second.program_);
  }

  // Public functions
  cl_int Build(const cl_device_id device, const std::string options) {
    return clBuildProgram(program_, 1, &device, options.c_str(), nullptr, nullptr);
  }
  std::string GetBuildInfo(const cl_device_id device) const {
    auto bytes = size_t{0};
    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &bytes);
    auto result = std::vector<char>(bytes);
    clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG, bytes, result.data(), nullptr);
    return std::string(result.data());
  }

  // Accessors to the private data-member
  cl_program operator()() const { return program_; }
  cl_program& operator()() { return program_; }
 private:
  size_t length_;
  std::vector<char> source_;
  const char* source_ptr_;
  cl_program program_;
};

// =================================================================================================

// C++11 version of cl_kernel
class Kernel {
 public:

  // Memory management
  Kernel(const Program program, const std::string &name):
    kernel_(clCreateKernel(program(), name.c_str(), nullptr)) { }
  ~Kernel() {
    clReleaseKernel(kernel_);
  }
  Kernel(const Kernel& other):
    kernel_(other.kernel_) {
    clRetainKernel(kernel_);
  }
  Kernel& operator=(Kernel other) {
    swap(*this, other);
    return *this;
  }
  void swap(Kernel& first, Kernel& second) {
    std::swap(first.kernel_, second.kernel_);
  }

  // Public functions
  template <typename T>
  cl_int SetArgument(const cl_uint index, const T value) {
    return clSetKernelArg(kernel_, index, sizeof(T), &value);
  }
  size_t GetLocalMemSize(const cl_device_id device) {
    auto bytes = size_t{0};
    clGetKernelWorkGroupInfo(kernel_, device, CL_KERNEL_LOCAL_MEM_SIZE, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetKernelWorkGroupInfo(kernel_, device, CL_KERNEL_LOCAL_MEM_SIZE, bytes, &result, nullptr);
    return result;
  }

  // Accessors to the private data-member
  cl_kernel operator()() const { return kernel_; }
  cl_kernel& operator()() { return kernel_; }
 private:
  cl_kernel kernel_;
};

// =================================================================================================

// C++11 version of cl_event
class Event {
 public:

  // Public functions
  size_t GetProfilingStart() const {
    auto bytes = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, bytes, &result, nullptr);
    return result;
  }
  size_t GetProfilingEnd() const {
    auto bytes = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, 0, nullptr, &bytes);
    auto result = size_t{0};
    clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, bytes, &result, nullptr);
    return result;
  }
  cl_int Wait() const {
    return clWaitForEvents(1, &event_);
  }

  // Accessors to the private data-member
  cl_event operator()() const { return event_; }
  cl_event& operator()() { return event_; }
 private:
  cl_event event_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_CLPP11_H_
#endif
