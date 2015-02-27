
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

#include "tuner/tuner.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>

namespace cltune {
// =================================================================================================

// Messages printed to stdout (in colours)
const std::string Tuner::kMessageFull    = "\x1b[32m[==========]\x1b[0m";
const std::string Tuner::kMessageHead    = "\x1b[32m[----------]\x1b[0m";
const std::string Tuner::kMessageRun     = "\x1b[32m[ RUN      ]\x1b[0m";
const std::string Tuner::kMessageInfo    = "\x1b[32m[   INFO   ]\x1b[0m";
const std::string Tuner::kMessageOK      = "\x1b[32m[       OK ]\x1b[0m";
const std::string Tuner::kMessageWarning = "\x1b[33m[  WARNING ]\x1b[0m";
const std::string Tuner::kMessageFailure = "\x1b[31m[   FAILED ]\x1b[0m";
const std::string Tuner::kMessageResult  = "\x1b[32m[ RESULT   ]\x1b[0m";
const std::string Tuner::kMessageBest    = "\x1b[35m[     BEST ]\x1b[0m";
  
// =================================================================================================

// Initializes the platform and device to the default 0
Tuner::Tuner():
    opencl_(new OpenCL(0, 0)),
    has_reference_(false),
    suppress_output_(false),
    argument_counter_(0) {
}

// Initializes with a custom platform and device
Tuner::Tuner(int platform_id, int device_id):
    opencl_(new OpenCL(platform_id, device_id)),
    has_reference_(false),
    suppress_output_(false),
    argument_counter_(0) {
}

// End of the tuner
Tuner::~Tuner() {
  for (auto &reference_output: reference_outputs_) {
    delete[] (int*)reference_output;
  }
  if (!suppress_output_) {
    fprintf(stdout, "\n%s End of the tuning process\n\n", kMessageFull.c_str());
  }
}

// =================================================================================================

// Loads the OpenCL source-code from a file and creates a new variable of type KernelInfo to store
// all the kernel-information.
int Tuner::AddKernel(const std::string &filename, const std::string &kernel_name,
                      const cl::NDRange &global, const cl::NDRange &local) {

  // Loads the source-code and adds the kernel
  auto source = LoadFile(filename);
  kernels_.push_back(KernelInfo(kernel_name, source));

  // Sets the global and local thread sizes
  auto id = kernels_.size() - 1;
  kernels_[id].set_global_base(global);
  kernels_[id].set_local_base(local);
  return id;
}

// =================================================================================================

// Sets the reference kernel (source-code location, kernel name, global/local thread-sizes) and
// sets a flag to indicate that there is now a reference. Calling this function again will simply
// overwrite the old reference.
void Tuner::SetReference(const std::string &filename, const std::string &kernel_name,
                         const cl::NDRange &global, const cl::NDRange &local) {
  has_reference_ = true;
  auto source = LoadFile(filename);
  reference_kernel_.reset(new KernelInfo(kernel_name, source));
  reference_kernel_->set_global_base(global);
  reference_kernel_->set_local_base(local);
}

// =================================================================================================

// Adds parameters for a kernel to tune. Also checks whether this parameter already exists.
void Tuner::AddParameter(const size_t id, const std::string parameter_name,
                         const std::initializer_list<int> values) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  if (kernels_[id].ParameterExists(parameter_name)) {
    throw Exception("Parameter already exists");
  }
  kernels_[id].AddParameter(parameter_name, values);
}

// =================================================================================================

// These functions forward their work (adding a modifier to global/local thread-sizes) to an object
// of KernelInfo class
void Tuner::MulGlobalSize(const size_t id, const StringRange range) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalMul);
}
void Tuner::DivGlobalSize(const size_t id, const StringRange range) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalDiv);
}
void Tuner::MulLocalSize(const size_t id, const StringRange range) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalMul);
}
void Tuner::DivLocalSize(const size_t id, const StringRange range) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalDiv);
}

// Adds a contraint to the list of constraints for a particular kernel. First checks whether the
// kernel exists and whether the parameters exist.
void Tuner::AddConstraint(const size_t id, KernelInfo::ConstraintFunction valid_if,
                          const std::vector<std::string> &parameters) {
  if (id >= kernels_.size()) { throw Exception("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!kernels_[id].ParameterExists(parameter)) { throw Exception("Invalid parameter"); }
  }
  kernels_[id].AddConstraint(valid_if, parameters);
}

// =================================================================================================

// Creates a new buffer of type Memory (containing both host and device data) based on a source
// vector of data. Then, upload it to the device and store the argument in a list.
template <typename T>
void Tuner::AddArgumentInput(std::vector<T> &source) {
  Memory<T> buffer(source.size(), opencl_, source);
  buffer.UploadToDevice();
  MemArgument argument = {argument_counter_++, source.size(), buffer.type, *buffer.device()};
  arguments_input_.push_back(argument);
}
template void Tuner::AddArgumentInput<int>(std::vector<int>&);
template void Tuner::AddArgumentInput<float>(std::vector<float>&);
template void Tuner::AddArgumentInput<double>(std::vector<double>&);

// As above, but now marked as output buffer
template <typename T>
void Tuner::AddArgumentOutput(std::vector<T> &source) {
  Memory<T> buffer(source.size(), opencl_, source);
  MemArgument argument = {argument_counter_++, source.size(), buffer.type, *buffer.device()};
  arguments_output_.push_back(argument);
}
template void Tuner::AddArgumentOutput<int>(std::vector<int>&);
template void Tuner::AddArgumentOutput<float>(std::vector<float>&);
template void Tuner::AddArgumentOutput<double>(std::vector<double>&);

// Sets a simple scalar value as an argument to the kernel
template <typename T>
void Tuner::AddArgumentScalar(const T argument) {
  arguments_scalar_.push_back({argument_counter_++, argument});
}
template void Tuner::AddArgumentScalar<int>(const int);
template void Tuner::AddArgumentScalar<float>(const float);
template void Tuner::AddArgumentScalar<double>(const double);

// =================================================================================================

// Starts the tuning process. First, the reference kernel is run if it exists (output results are
// automatically verified with respect to this reference run). Next, all permutations of all tuning-
// parameters are computed for each kernel and those kernels are run. Their timing-results are
// collected and stored into the tuning_results_ vector.
void Tuner::Tune() {

  // Runs the reference kernel if it is defined
  if (has_reference_) {
    PrintHeader("Testing reference "+reference_kernel_->name());
    RunKernel(reference_kernel_->source(), *reference_kernel_, 0, 1);
    StoreReferenceOutput();
  }
  
  // Iterates over all tunable kernels
  for (auto &kernel: kernels_) {
    PrintHeader("Testing kernel "+kernel.name());

    // If there are no tuning parameters, simply run the kernel and store the results
    if (kernel.parameters().size() == 0) {

        // Compiles and runs the kernel
      auto tuning_result = RunKernel(kernel.source(), kernel, 0, 1);
      tuning_result.status = VerifyOutput();

      // Stores the result of the tuning
      tuning_results_.push_back(tuning_result);

    // Else: there are tuning parameters to iterate over
    } else {

      // Computes the permutations of all parameters and pass them to a (smart) search algorithm
      kernel.SetConfigurations();

      // Iterates over all possible configurations (the permutations of the tuning parameters)
      auto num_configurations = kernel.configurations().size();
      for (auto p=0; p<num_configurations; ++p) {
        auto permutation = kernel.configurations()[p];

        // Adds the parameters to the source-code string as defines
        auto source = std::string{};
        for (auto &config: permutation) {
          source += config.GetDefine();
        }
        source += kernel.source();

        // Updates the local range with the parameter values
        kernel.ComputeRanges(permutation);

        // Compiles and runs the kernel
        auto tuning_result = RunKernel(source, kernel, p, num_configurations);
        tuning_result.status = VerifyOutput();

        // Stores the parameters and the timing-result
        tuning_result.configuration = permutation;
        tuning_results_.push_back(tuning_result);
        if (!tuning_result.status) { PrintResult(stdout, tuning_result, kMessageWarning); }
      }
    }
  }
}

// =================================================================================================

// Iterates over all tuning results and prints each parameter configuration and the corresponding
// timing-results. Printing is to stdout.
double Tuner::PrintToScreen() const {

  // Finds the best result
  TunerResult best_result;
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Aborts if there was no best time found
  if (best_time == std::numeric_limits<double>::max()) {
    PrintHeader("No tuner results found");
    return 0.0;
  }

  // Prints all valid results and the one with the lowest execution time
  PrintHeader("Printing results to stdout");
  for (auto &tuning_result: tuning_results_) {
    if (tuning_result.status) {
      PrintResult(stdout, tuning_result, kMessageResult);
    }
  }
  PrintHeader("Printing best result to stdout");
  PrintResult(stdout, best_result, kMessageBest);
  return best_time;
}

// Same as PrintToScreen, but now outputs into a file and does not mark the best-case
void Tuner::PrintToFile(const std::string &filename) const {
  PrintHeader("Printing results to file: "+filename);
  auto file = fopen(filename.c_str(), "w");
  std::vector<std::string> processed_kernels;
  for (auto &tuning_result: tuning_results_) {
    if (tuning_result.status) {

      // Checks whether this is a kernel which hasn't been encountered yet
      auto new_kernel = true;
      for (auto &kernel_name: processed_kernels) {
        if (kernel_name == tuning_result.kernel_name) { new_kernel = false; break; }
      }
      processed_kernels.push_back(tuning_result.kernel_name);

      // Prints the header in case of a new kernel name
      if (new_kernel) {
        fprintf(file, "name;time;threads;");
        for (auto &setting: tuning_result.configuration) {
          fprintf(file, "%s;", setting.name.c_str());
        }
        fprintf(file, "\n");
      }

      // Prints an entry to file
      fprintf(file, "%s;", tuning_result.kernel_name.c_str());
      fprintf(file, "%.2lf;", tuning_result.time);
      fprintf(file, "%lu;", tuning_result.threads);
      for (auto &setting: tuning_result.configuration) {
        fprintf(file, "%d;", setting.value);
      }
      fprintf(file, "\n");
    }
  }
  fclose(file);
}

// Set the flag to suppress output to true. Note that this cannot be undone.
void Tuner::SuppressOutput() {
  suppress_output_ = true;
}

// End of the public methods
// =================================================================================================
// =================================================================================================

// Compiles the kernel and checks for OpenCL error messages, sets all output buffers to zero,
// launches the kernel, and collects the timing information.
Tuner::TunerResult Tuner::RunKernel(const std::string &source, const KernelInfo &kernel,
                                    const int configuration_id, const int num_configurations) {

  // Collects the source
  cl::Program::Sources sources;
  sources.push_back({source.c_str(), source.length()});

  // Compiles the kernel and prints the compiler errors/warnings
  cl::Program program(opencl_->context(), sources);
  auto options = std::string{};
  auto status = program.build({opencl_->device()}, options.c_str());
  if (status == CL_BUILD_PROGRAM_FAILURE || status == CL_INVALID_BINARY) {
    auto message = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(opencl_->device());
    throw Exception("OpenCL compiler error/warning:\n" + message);
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
      default: throw Exception("Unsupported reference output data-type");
    }
  }

  // Sets the kernel and its arguments
  auto tune_kernel = cl::Kernel(program, kernel.name().c_str());
  for (auto &i: arguments_input_)  { tune_kernel.setArg(i.index, i.buffer); }
  for (auto &i: arguments_output_) { tune_kernel.setArg(i.index, i.buffer); }
  for (auto &i: arguments_scalar_) { tune_kernel.setArg(i.first, i.second); }

  // Sets the global and local thread-sizes
  auto global = kernel.global();
  auto local = kernel.local();

  // Verifies the global/local thread-sizes against device properties
  auto local_threads = opencl_->VerifyThreadSizes(global, local);

  // Obtains and verifies the local memory usage of the kernel
  auto local_memory = static_cast<size_t>(0);
  status = tune_kernel.getWorkGroupInfo(opencl_->device(), CL_KERNEL_LOCAL_MEM_SIZE, &local_memory);
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Get kernel information error", status); }
  opencl_->VerifyLocalMemory(local_memory);

  // Prepares the kernel
  status = opencl_->queue().finish();
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Command queue error", status); }

  // Runs the kernel (this is the timed part)
  fprintf(stdout, "%s Running %s\n", kMessageRun.c_str(), kernel.name().c_str());
  std::vector<cl::Event> events(kNumRuns);
  for (auto t=0; t<kNumRuns; ++t) {
    status = opencl_->queue().enqueueNDRangeKernel(tune_kernel, cl::NullRange, global, local, NULL, &events[t]);
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
  fprintf(stdout, "%s Completed %s (%.0lf ms) - %d out of %d\n",
          kMessageOK.c_str(), kernel.name().c_str(), elapsed_time,
          configuration_id+1, num_configurations);

  // Computes the result of the tuning
  TunerResult result = {kernel.name(), elapsed_time, local_threads, false, {}};
  return result;
}

// =================================================================================================

// Creates a new array of zeroes and copies it to the target OpenCL buffer
template <typename T> 
void Tuner::ResetMemArgument(MemArgument &argument) {

  // Create an array with zeroes
  std::vector<T> buffer(argument.size, static_cast<T>(0));

  // Copy the new array to the OpenCL buffer on the device
  auto bytes = sizeof(T)*argument.size;
  auto status = opencl_->queue().enqueueWriteBuffer(argument.buffer, CL_TRUE, 0, bytes,
                                                    buffer.data());
  if (status != CL_SUCCESS) { throw OpenCL::Exception("Write buffer error", status); }
}

// =================================================================================================

// Loops over all reference outputs, creates per output a new host buffer and copies the OpenCL
// buffer from the device onto the host. This function is specialised for different data-types.
void Tuner::StoreReferenceOutput() {
  reference_outputs_.clear();
  for (auto &output_buffer: arguments_output_) {
    switch (output_buffer.type) {
      case MemType::kInt: DownloadReference<int>(output_buffer); break;
      case MemType::kFloat: DownloadReference<float>(output_buffer); break;
      case MemType::kDouble: DownloadReference<double>(output_buffer); break;
      default: throw Exception("Unsupported reference output data-type");
    }
  }
}
template <typename T> void Tuner::DownloadReference(const MemArgument &device_buffer) {
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
bool Tuner::VerifyOutput() {
  auto status = true;
  if (has_reference_) {
    for (auto i=0; i<arguments_output_.size(); ++i) {
      auto output_buffer = arguments_output_[i];
      switch (output_buffer.type) {
        case MemType::kInt: status &= DownloadAndCompare<int>(output_buffer, i); break;
        case MemType::kFloat: status &= DownloadAndCompare<float>(output_buffer, i); break;
        case MemType::kDouble: status &= DownloadAndCompare<double>(output_buffer, i); break;
        default: throw Exception("Unsupported output data-type");
      }
    }
  }
  return status;
}

// See above comment
template <typename T>
bool Tuner::DownloadAndCompare(const MemArgument &device_buffer, const size_t i) {
  auto l2_norm = 0.0;

  // Downloads the results to the host
  std::vector<T> host_buffer(device_buffer.size);
  auto bytes = sizeof(T)*device_buffer.size;
  opencl_->queue().enqueueReadBuffer(device_buffer.buffer, CL_TRUE, 0, bytes, host_buffer.data());

  // Compares the results (L2 norm)
  T* reference_output = (T*)reference_outputs_[i];
  for (auto j=0; j<device_buffer.size; ++j) {
    l2_norm += fabs((double)reference_output[j] - (double)host_buffer[j]);
  }

  // Verifies if everything was OK, if not: print the L2 norm
  // TODO: Implement a choice of comparisons for the client to choose from
  if (std::isnan(l2_norm) || l2_norm > kMaxL2Norm) {
    fprintf(stderr, "%s Results differ: L2 norm is %6.2e\n", kMessageWarning.c_str(), l2_norm);
    return false;
  }
  return true;
}

// =================================================================================================

// Prints a result by looping over all its configuration parameters
void Tuner::PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const {
  fprintf(fp, "%s %s; ", message.c_str(), result.kernel_name.c_str());
  fprintf(fp, "%6.0lf ms;", result.time);
  for (auto &setting: result.configuration) {
    fprintf(fp, "%9s;", setting.GetConfig().c_str());
  }
  fprintf(fp, "\n");
}

// =================================================================================================

// Loads a file into a stringstream and returns the result as a string
std::string Tuner::LoadFile(const std::string &filename) {
  std::ifstream file(filename);
  if (file.fail()) { throw Exception("Could not open kernel file: "+filename); }
  std::stringstream file_contents;
  file_contents << file.rdbuf();
  return file_contents.str();
}

// =================================================================================================

// Converts a C++ string to a C string and print it out with nice formatting
void Tuner::PrintHeader(const std::string &header_name) const {
  if (!suppress_output_) {
    fprintf(stdout, "\n%s %s\n", kMessageHead.c_str(), header_name.c_str());
  }
}

// =================================================================================================
} // namespace cltune

