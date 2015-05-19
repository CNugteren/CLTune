
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the Tuner class (see the header for information about the class).
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

#include "cltune.h"

// The corresponding header file
#include "internal/tuner_impl.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <regex>

namespace cltune {
// =================================================================================================

// Messages printed to stdout (in colours)
const std::string TunerImpl::kMessageFull    = "\x1b[32m[==========]\x1b[0m";
const std::string TunerImpl::kMessageHead    = "\x1b[32m[----------]\x1b[0m";
const std::string TunerImpl::kMessageRun     = "\x1b[32m[ RUN      ]\x1b[0m";
const std::string TunerImpl::kMessageInfo    = "\x1b[32m[   INFO   ]\x1b[0m";
const std::string TunerImpl::kMessageOK      = "\x1b[32m[       OK ]\x1b[0m";
const std::string TunerImpl::kMessageWarning = "\x1b[33m[  WARNING ]\x1b[0m";
const std::string TunerImpl::kMessageFailure = "\x1b[31m[   FAILED ]\x1b[0m";
const std::string TunerImpl::kMessageResult  = "\x1b[32m[ RESULT   ]\x1b[0m";
const std::string TunerImpl::kMessageBest    = "\x1b[35m[     BEST ]\x1b[0m";
  
// =================================================================================================

// Initializes the platform and device to the default 0
TunerImpl::TunerImpl():
    opencl_(new OpenCL(0, 0)),
    has_reference_(false),
    suppress_output_(false),
    output_search_process_(false),
    search_log_filename_(std::string{}),
    search_method_(SearchMethod::FullSearch),
    search_args_(0),
    argument_counter_(0) {
}

// Initializes with a custom platform and device
TunerImpl::TunerImpl(size_t platform_id, size_t device_id):
    opencl_(new OpenCL(platform_id, device_id)),
    has_reference_(false),
    suppress_output_(false),
    output_search_process_(false),
    search_log_filename_(std::string{}),
    search_method_(SearchMethod::FullSearch),
    search_args_(0),
    argument_counter_(0) {
}

// End of the tuner
TunerImpl::~TunerImpl() {
  for (auto &reference_output: reference_outputs_) {
    delete[] (int*)reference_output;
  }
  if (!suppress_output_) {
    fprintf(stdout, "\n%s End of the tuning process\n\n", kMessageFull.c_str());
  }
}

// =================================================================================================

// Compiles the kernel and checks for OpenCL error messages, sets all output buffers to zero,
// launches the kernel, and collects the timing information.
TunerImpl::TunerResult TunerImpl::RunKernel(const std::string &source, const KernelInfo &kernel,
                                            const size_t configuration_id,
                                            const size_t num_configurations) {

  // Removes the use of C++11 string literals (if any) from the kernel source code
  auto string_literal_start = std::regex{"R\"\\("};
  auto string_literal_end = std::regex{"\\)\";"};
  auto processed_source = std::regex_replace(source, string_literal_start, "");
  processed_source = std::regex_replace(processed_source, string_literal_end, "");

  // Collects the source
  cl::Program::Sources sources;
  sources.push_back({processed_source.c_str(), processed_source.length()});

  // Compiles the kernel and prints the compiler errors/warnings
  cl::Program program(opencl_->context(), sources);
  auto options = std::string{};
  auto status = program.build({opencl_->device()}, options.c_str());
  if (status == CL_BUILD_PROGRAM_FAILURE || status == CL_INVALID_BINARY) {
    auto message = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(opencl_->device());
    throw std::runtime_error("OpenCL compiler error/warning:\n" + message);
  }
  if (status != CL_SUCCESS) {
    throw OpenCL::Exception("Program build error", status);
  }

  // Sets the output buffer(s) to zero
  for (auto &output: arguments_output_) {
    switch (output.type) {
      case MemType::kInt: ResetMemArgument<int>(output); break;
      case MemType::kFloat: ResetMemArgument<float>(output); break;
      case MemType::kDouble: ResetMemArgument<double>(output); break;
      case MemType::kFloat2: ResetMemArgument<float2>(output); break;
      case MemType::kDouble2: ResetMemArgument<double2>(output); break;
      default: throw std::runtime_error("Unsupported reference output data-type");
    }
  }

  // Sets the kernel and its arguments
  auto tune_kernel = cl::Kernel(program, kernel.name().c_str());
  for (auto &i: arguments_input_)  { tune_kernel.setArg(i.index, i.buffer); }
  for (auto &i: arguments_output_) { tune_kernel.setArg(i.index, i.buffer); }
  for (auto &i: arguments_int_) { tune_kernel.setArg(i.first, i.second); }
  for (auto &i: arguments_size_t_) { tune_kernel.setArg(i.first, i.second); }
  for (auto &i: arguments_float_) { tune_kernel.setArg(i.first, i.second); }
  for (auto &i: arguments_double_) { tune_kernel.setArg(i.first, i.second); }
  for (auto &i: arguments_float2_) { tune_kernel.setArg(i.first, i.second); }
  for (auto &i: arguments_double2_) { tune_kernel.setArg(i.first, i.second); }

  // Sets the global and local thread-sizes
  auto global = kernel.global();
  auto local = kernel.local();
  cl::NDRange global_temp;
  cl::NDRange local_temp;
  switch (global.size()) {
    case 1:
      global_temp = cl::NDRange(global[0]);
      local_temp = cl::NDRange(local[0]);
      break;
    case 2:
      global_temp = cl::NDRange(global[0], global[1]);
      local_temp = cl::NDRange(local[0], local[1]);
      break;
    case 3:
      global_temp = cl::NDRange(global[0], global[1], global[2]);
      local_temp = cl::NDRange(local[0], local[1], local[2]);
      break;
  }

  // In case of an exception, skip this run
  try {

    // Obtains and verifies the local memory usage of the kernel
    auto local_memory = static_cast<size_t>(0);
    status = tune_kernel.getWorkGroupInfo(opencl_->device(), CL_KERNEL_LOCAL_MEM_SIZE, &local_memory);
    if (status != CL_SUCCESS) { throw OpenCL::Exception("Get kernel information error", status); }
    if (!opencl_->ValidLocalMemory(local_memory)) { throw std::runtime_error("Using too much local memory"); }

    // Prepares the kernel
    status = opencl_->queue().finish();
    if (status != CL_SUCCESS) { throw OpenCL::Exception("Command queue error", status); }

    // Runs the kernel (this is the timed part)
    fprintf(stdout, "%s Running %s\n", kMessageRun.c_str(), kernel.name().c_str());
    std::vector<cl::Event> events(kNumRuns);
    for (auto t=0; t<kNumRuns; ++t) {
      status = opencl_->queue().enqueueNDRangeKernel(tune_kernel, cl::NullRange, global_temp, local_temp, NULL, &events[t]);
      if (status != CL_SUCCESS) { throw OpenCL::Exception("Kernel launch error", status); }
      status = events[t].wait();
      if (status != CL_SUCCESS) {
        fprintf(stdout, "%s Kernel %s failed\n", kMessageFailure.c_str(), kernel.name().c_str());
        throw OpenCL::Exception("Kernel error", status);
      }
    }
    opencl_->queue().finish();

    // Collects the timing information
    auto elapsed_time = std::numeric_limits<double>::max();
    for (auto t=0; t<kNumRuns; ++t) {
      auto start_time = events[t].getProfilingInfo<CL_PROFILING_COMMAND_START>(&status);
      auto end_time = events[t].getProfilingInfo<CL_PROFILING_COMMAND_END>(&status);
      elapsed_time = std::min(elapsed_time, (end_time - start_time) / (1000.0 * 1000.0));
    }

    // Prints diagnostic information
    fprintf(stdout, "%s Completed %s (%.0lf ms) - %lu out of %lu\n",
            kMessageOK.c_str(), kernel.name().c_str(), elapsed_time,
            configuration_id+1, num_configurations);

    // Computes the result of the tuning
    auto local_threads = opencl_->GetLocalSize(global, local);
    TunerResult result = {kernel.name(), elapsed_time, local_threads, false, {}};
    return result;
  }

  // There was an exception, now return an invalid tuner results
  catch(std::exception& e) {
    TunerResult result = {kernel.name(), std::numeric_limits<double>::max(), 0, false, {}};
    return result;
  }
}

// =================================================================================================

// Creates a new array of zeroes and copies it to the target OpenCL buffer
template <typename T> 
void TunerImpl::ResetMemArgument(MemArgument &argument) {

  // Create an array with zeroes
  std::vector<T> buffer(argument.size, T{0});

  // Copy the new array to the OpenCL buffer on the device
  auto bytes = sizeof(T)*argument.size;
  auto status = opencl_->queue().enqueueWriteBuffer(argument.buffer, CL_TRUE, 0, bytes,
                                                    buffer.data());
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Write buffer error", status); }
}

// =================================================================================================

// Loops over all reference outputs, creates per output a new host buffer and copies the OpenCL
// buffer from the device onto the host. This function is specialised for different data-types.
void TunerImpl::StoreReferenceOutput() {
  reference_outputs_.clear();
  for (auto &output_buffer: arguments_output_) {
    switch (output_buffer.type) {
      case MemType::kInt: DownloadReference<int>(output_buffer); break;
      case MemType::kFloat: DownloadReference<float>(output_buffer); break;
      case MemType::kDouble: DownloadReference<double>(output_buffer); break;
      case MemType::kFloat2: DownloadReference<float2>(output_buffer); break;
      case MemType::kDouble2: DownloadReference<double2>(output_buffer); break;
      default: throw std::runtime_error("Unsupported reference output data-type");
    }
  }
}
template <typename T> void TunerImpl::DownloadReference(const MemArgument &device_buffer) {
  T* host_buffer = new T[device_buffer.size];
  auto bytes = sizeof(T)*device_buffer.size;
  opencl_->queue().enqueueReadBuffer(device_buffer.buffer, CL_TRUE, 0, bytes, host_buffer);
  reference_outputs_.push_back(host_buffer);
}

// =================================================================================================

// In case there is a reference kernel, this function loops over all outputs, creates per output a
// new host buffer and copies the OpenCL buffer from the device onto the host. Following, it
// compares the results to the reference output. This function is specialised for different
// data-types. These functions return "true" if everything is OK, and "false" if there is a warning.
bool TunerImpl::VerifyOutput() {
  auto status = true;
  if (has_reference_) {
    auto i = 0;
    for (auto &output_buffer: arguments_output_) {
      switch (output_buffer.type) {
        case MemType::kInt: status &= DownloadAndCompare<int>(output_buffer, i); break;
        case MemType::kFloat: status &= DownloadAndCompare<float>(output_buffer, i); break;
        case MemType::kDouble: status &= DownloadAndCompare<double>(output_buffer, i); break;
        case MemType::kFloat2: status &= DownloadAndCompare<float2>(output_buffer, i); break;
        case MemType::kDouble2: status &= DownloadAndCompare<double2>(output_buffer, i); break;
        default: throw std::runtime_error("Unsupported output data-type");
      }
      ++i;
    }
  }
  return status;
}

// See above comment
template <typename T>
bool TunerImpl::DownloadAndCompare(const MemArgument &device_buffer, const size_t i) {
  auto l2_norm = 0.0;

  // Downloads the results to the host
  std::vector<T> host_buffer(device_buffer.size);
  auto bytes = sizeof(T)*device_buffer.size;
  opencl_->queue().enqueueReadBuffer(device_buffer.buffer, CL_TRUE, 0, bytes, host_buffer.data());

  // Compares the results (L2 norm)
  T* reference_output = (T*)reference_outputs_[i];
  for (auto j=0UL; j<device_buffer.size; ++j) {
    l2_norm += AbsoluteDifference(reference_output[j], host_buffer[j]);
  }

  // Verifies if everything was OK, if not: print the L2 norm
  // TODO: Implement a choice of comparisons for the client to choose from
  if (std::isnan(l2_norm) || l2_norm > kMaxL2Norm) {
    fprintf(stderr, "%s Results differ: L2 norm is %6.2e\n", kMessageWarning.c_str(), l2_norm);
    return false;
  }
  return true;
}

// Computes the absolute difference
template <typename T>
double TunerImpl::AbsoluteDifference(const T reference, const T result) {
  return fabs(static_cast<double>(reference) - static_cast<double>(result));
}
template <> double TunerImpl::AbsoluteDifference(const float2 reference, const float2 result) {
  auto real = fabs(static_cast<double>(reference.real()) - static_cast<double>(result.real()));
  auto imag = fabs(static_cast<double>(reference.imag()) - static_cast<double>(result.imag()));
  return real + imag;
}
template <> double TunerImpl::AbsoluteDifference(const double2 reference, const double2 result) {
  auto real = fabs(reference.real() - result.real());
  auto imag = fabs(reference.imag() - result.imag());
  return real + imag;
}

// =================================================================================================

// Prints a result by looping over all its configuration parameters
void TunerImpl::PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const {
  fprintf(fp, "%s %s; ", message.c_str(), result.kernel_name.c_str());
  fprintf(fp, "%6.0lf ms;", result.time);
  for (auto &setting: result.configuration) {
    fprintf(fp, "%9s;", setting.GetConfig().c_str());
  }
  fprintf(fp, "\n");
}

// =================================================================================================

// Loads a file into a stringstream and returns the result as a string
std::string TunerImpl::LoadFile(const std::string &filename) {
  std::ifstream file(filename);
  if (file.fail()) { throw std::runtime_error("Could not open kernel file: "+filename); }
  std::stringstream file_contents;
  file_contents << file.rdbuf();
  return file_contents.str();
}

// =================================================================================================

// Converts a C++ string to a C string and print it out with nice formatting
void TunerImpl::PrintHeader(const std::string &header_name) const {
  if (!suppress_output_) {
    fprintf(stdout, "\n%s %s\n", kMessageHead.c_str(), header_name.c_str());
  }
}

// =================================================================================================
} // namespace cltune

