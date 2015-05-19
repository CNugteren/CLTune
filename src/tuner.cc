
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
#include "cltune/tuner.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <regex>

#include "cltune/searchers/full_search.h"
#include "cltune/searchers/random_search.h"
#include "cltune/searchers/annealing.h"
#include "cltune/searchers/pso.h"

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
    search_method_(Tuner::SearchMethod::FullSearch),
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
    search_method_(Tuner::SearchMethod::FullSearch),
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

Tuner::Tuner():
    pimpl_(new TunerImpl()) {
}
Tuner::Tuner(size_t platform_id, size_t device_id):
    pimpl_(new TunerImpl(platform_id, device_id)) {
}
Tuner::~Tuner() {
}

// =================================================================================================

// Loads the OpenCL source-code from a file and creates a new variable of type KernelInfo to store
// all the kernel-information.
size_t Tuner::AddKernel(const std::vector<std::string> &filenames, const std::string &kernel_name,
                        const IntRange &global, const IntRange &local) {

  // Loads the source-code and adds the kernel
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl_->LoadFile(filename);
  }
  pimpl_->kernels_.push_back(KernelInfo(kernel_name, source, pimpl_->opencl_));

  // Sets the global and local thread sizes
  auto id = pimpl_->kernels_.size() - 1;
  pimpl_->kernels_[id].set_global_base(global);
  pimpl_->kernels_[id].set_local_base(local);
  return id;
}

// =================================================================================================

// Sets the reference kernel (source-code location, kernel name, global/local thread-sizes) and
// sets a flag to indicate that there is now a reference. Calling this function again will simply
// overwrite the old reference.
void Tuner::SetReference(const std::vector<std::string> &filenames, const std::string &kernel_name,
                         const IntRange &global, const IntRange &local) {
  pimpl_->has_reference_ = true;
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl_->LoadFile(filename);
  }
  pimpl_->reference_kernel_.reset(new KernelInfo(kernel_name, source, pimpl_->opencl_));
  pimpl_->reference_kernel_->set_global_base(global);
  pimpl_->reference_kernel_->set_local_base(local);
}

// =================================================================================================

// Adds parameters for a kernel to tune. Also checks whether this parameter already exists.
void Tuner::AddParameter(const size_t id, const std::string &parameter_name,
                         const std::initializer_list<size_t> &values) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  if (pimpl_->kernels_[id].ParameterExists(parameter_name)) {
    throw std::runtime_error("Parameter already exists");
  }
  pimpl_->kernels_[id].AddParameter(parameter_name, values);
}

// =================================================================================================

// These functions forward their work (adding a modifier to global/local thread-sizes) to an object
// of KernelInfo class
void Tuner::MulGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl_->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalMul);
}
void Tuner::DivGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl_->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalDiv);
}
void Tuner::MulLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl_->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalMul);
}
void Tuner::DivLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl_->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalDiv);
}

// Adds a contraint to the list of constraints for a particular kernel. First checks whether the
// kernel exists and whether the parameters exist.
void Tuner::AddConstraint(const size_t id, ConstraintFunction valid_if,
                          const std::vector<std::string> &parameters) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl_->kernels_[id].ParameterExists(parameter)) { throw std::runtime_error("Invalid parameter"); }
  }
  pimpl_->kernels_[id].AddConstraint(valid_if, parameters);
}

// As above, but for the local memory usage
void Tuner::SetLocalMemoryUsage(const size_t id, LocalMemoryFunction amount,
                                const std::vector<std::string> &parameters) {
  if (id >= pimpl_->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl_->kernels_[id].ParameterExists(parameter)) { throw std::runtime_error("Invalid parameter"); }
  }
 pimpl_->kernels_[id].SetLocalMemoryUsage(amount, parameters);
}


// =================================================================================================

// Creates a new buffer of type Memory (containing both host and device data) based on a source
// vector of data. Then, upload it to the device and store the argument in a list.
template <typename T>
void Tuner::AddArgumentInput(const std::vector<T> &source) {
  auto buffer = Memory<T>{source.size(), pimpl_->opencl_->queue(), pimpl_->opencl_->context(),
                          CL_MEM_READ_ONLY, source};
  buffer.UploadToDevice();
  TunerImpl::MemArgument argument = {pimpl_->argument_counter_++, source.size(),
                                     buffer.type, *buffer.device()};
  pimpl_->arguments_input_.push_back(argument);
}
template void Tuner::AddArgumentInput<int>(const std::vector<int>&);
template void Tuner::AddArgumentInput<float>(const std::vector<float>&);
template void Tuner::AddArgumentInput<double>(const std::vector<double>&);
template void Tuner::AddArgumentInput<float2>(const std::vector<float2>&);
template void Tuner::AddArgumentInput<double2>(const std::vector<double2>&);

// As above, but now marked as output buffer
template <typename T>
void Tuner::AddArgumentOutput(const std::vector<T> &source) {
  auto buffer = Memory<T>{source.size(), pimpl_->opencl_->queue(), pimpl_->opencl_->context(),
                          CL_MEM_READ_WRITE, source};
  TunerImpl::MemArgument argument = {pimpl_->argument_counter_++, source.size(),
                                     buffer.type, *buffer.device()};
  pimpl_->arguments_output_.push_back(argument);
}
template void Tuner::AddArgumentOutput<int>(const std::vector<int>&);
template void Tuner::AddArgumentOutput<float>(const std::vector<float>&);
template void Tuner::AddArgumentOutput<double>(const std::vector<double>&);
template void Tuner::AddArgumentOutput<float2>(const std::vector<float2>&);
template void Tuner::AddArgumentOutput<double2>(const std::vector<double2>&);

// Sets a scalar value as an argument to the kernel
template <> void Tuner::AddArgumentScalar<int>(const int argument) {
  pimpl_->arguments_int_.push_back({pimpl_->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<size_t>(const size_t argument) {
  pimpl_->arguments_size_t_.push_back({pimpl_->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<float>(const float argument) {
  pimpl_->arguments_float_.push_back({pimpl_->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<double>(const double argument) {
  pimpl_->arguments_double_.push_back({pimpl_->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<float2>(const float2 argument) {
  pimpl_->arguments_float2_.push_back({pimpl_->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<double2>(const double2 argument) {
  pimpl_->arguments_double2_.push_back({pimpl_->argument_counter_++, argument});
}

// =================================================================================================

// Use full search as a search strategy. This is the default method.
void Tuner::UseFullSearch() {
  pimpl_->search_method_ = SearchMethod::FullSearch;
}

// Use random search as a search strategy.
void Tuner::UseRandomSearch(const double fraction) {
  pimpl_->search_method_ = SearchMethod::RandomSearch;
  pimpl_->search_args_.push_back(fraction);
}

// Use simulated annealing as a search strategy.
void Tuner::UseAnnealing(const double fraction, const double max_temperature) {
  pimpl_->search_method_ = SearchMethod::Annealing;
  pimpl_->search_args_.push_back(fraction);
  pimpl_->search_args_.push_back(max_temperature);
}

// Use PSO as a search strategy.
void Tuner::UsePSO(const double fraction, const size_t swarm_size, const double influence_global,
                       const double influence_local, const double influence_random) {
  pimpl_->search_method_ = SearchMethod::PSO;
  pimpl_->search_args_.push_back(fraction);
  pimpl_->search_args_.push_back(static_cast<double>(swarm_size));
  pimpl_->search_args_.push_back(influence_global);
  pimpl_->search_args_.push_back(influence_local);
  pimpl_->search_args_.push_back(influence_random);
}


// Output the search process to a file. This is disabled per default.
void Tuner::OutputSearchLog(const std::string &filename) {
  pimpl_->output_search_process_ = true;
  pimpl_->search_log_filename_ = filename;
}

// =================================================================================================

// Starts the tuning process. First, the reference kernel is run if it exists (output results are
// automatically verified with respect to this reference run). Next, all permutations of all tuning-
// parameters are computed for each kernel and those kernels are run. Their timing-results are
// collected and stored into the tuning_results_ vector.
void Tuner::Tune() {

  // Runs the reference kernel if it is defined
  if (pimpl_->has_reference_) {
    pimpl_->PrintHeader("Testing reference "+pimpl_->reference_kernel_->name());
    pimpl_->RunKernel(pimpl_->reference_kernel_->source(), *pimpl_->reference_kernel_, 0, 1);
    pimpl_->StoreReferenceOutput();
  }
  
  // Iterates over all tunable kernels
  for (auto &kernel: pimpl_->kernels_) {
    pimpl_->PrintHeader("Testing kernel "+kernel.name());

    // If there are no tuning parameters, simply run the kernel and store the results
    if (kernel.parameters().size() == 0) {

        // Compiles and runs the kernel
      auto tuning_result = pimpl_->RunKernel(kernel.source(), kernel, 0, 1);
      tuning_result.status = pimpl_->VerifyOutput();

      // Stores the result of the tuning
      pimpl_->tuning_results_.push_back(tuning_result);

    // Else: there are tuning parameters to iterate over
    } else {

      // Computes the permutations of all parameters and pass them to a (smart) search algorithm
      kernel.SetConfigurations();

      // Creates the selected search algorithm
      std::unique_ptr<Searcher> search;
      switch (pimpl_->search_method_) {
        case SearchMethod::FullSearch:
          search.reset(new FullSearch{kernel.configurations()});
          break;
        case SearchMethod::RandomSearch:
          search.reset(new RandomSearch{kernel.configurations(), pimpl_->search_args_[0]});
          break;
        case SearchMethod::Annealing:
          search.reset(new Annealing{kernel.configurations(), pimpl_->search_args_[0], pimpl_->search_args_[1]});
          break;
        case SearchMethod::PSO:
          search.reset(new PSO{kernel.configurations(), kernel.parameters(), pimpl_->search_args_[0],
                               static_cast<size_t>(pimpl_->search_args_[1]), pimpl_->search_args_[2],
                               pimpl_->search_args_[3], pimpl_->search_args_[4]});
          break;
      }

      // Iterates over all possible configurations (the permutations of the tuning parameters)
      for (auto p=0UL; p<search->NumConfigurations(); ++p) {
        auto permutation = search->GetConfiguration();

        // Adds the parameters to the source-code string as defines
        auto source = std::string{};
        for (auto &config: permutation) {
          source += config.GetDefine();
        }
        source += kernel.source();

        // Updates the local range with the parameter values
        kernel.ComputeRanges(permutation);

        // Compiles and runs the kernel
        auto tuning_result = pimpl_->RunKernel(source, kernel, p, search->NumConfigurations());
        tuning_result.status = pimpl_->VerifyOutput();

        // Gives timing feedback to the search algorithm and calculate the next index
        search->PushExecutionTime(tuning_result.time);
        search->CalculateNextIndex();

        // Stores the parameters and the timing-result
        tuning_result.configuration = permutation;
        pimpl_->tuning_results_.push_back(tuning_result);
        if (!tuning_result.status) {
          pimpl_->PrintResult(stdout, tuning_result, pimpl_->kMessageWarning);
        }
      }

      // Prints a log of the searching process. This is disabled per default, but can be enabled
      // using the "OutputSearchLog" function.
      if (pimpl_->output_search_process_) {
        auto file = fopen(pimpl_->search_log_filename_.c_str(), "w");
        search->PrintLog(file);
        fclose(file);
      }
    }
  }
}

// =================================================================================================

// Iterates over all tuning results and prints each parameter configuration and the corresponding
// timing-results. Printing is to stdout.
double Tuner::PrintToScreen() const {

  // Finds the best result
  auto best_result = pimpl_->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl_->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Aborts if there was no best time found
  if (best_time == std::numeric_limits<double>::max()) {
    pimpl_->PrintHeader("No tuner results found");
    return 0.0;
  }

  // Prints all valid results and the one with the lowest execution time
  pimpl_->PrintHeader("Printing results to stdout");
  for (auto &tuning_result: pimpl_->tuning_results_) {
    if (tuning_result.status) {
      pimpl_->PrintResult(stdout, tuning_result, pimpl_->kMessageResult);
    }
  }
  pimpl_->PrintHeader("Printing best result to stdout");
  pimpl_->PrintResult(stdout, best_result, pimpl_->kMessageBest);

  // Return the best time
  return best_time;
}

// Prints the best result in a neatly formatted C++ database format to screen
void Tuner::PrintFormatted() const {

  // Finds the best result
  auto best_result = pimpl_->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl_->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Prints the best result in C++ database format
  auto count = 0UL;
  pimpl_->PrintHeader("Printing best result in database format to stdout");
  fprintf(stdout, "{ \"%s\", { ", pimpl_->opencl_->device_name().c_str());
  for (auto &setting: best_result.configuration) {
    fprintf(stdout, "%s", setting.GetDatabase().c_str());
    if (count < best_result.configuration.size()-1) {
      fprintf(stdout, ", ");
    }
    count++;
  }
  fprintf(stdout, " } }\n");
}

// Same as PrintToScreen, but now outputs into a file and does not mark the best-case
void Tuner::PrintToFile(const std::string &filename) const {
  pimpl_->PrintHeader("Printing results to file: "+filename);
  auto file = fopen(filename.c_str(), "w");
  std::vector<std::string> processed_kernels;
  for (auto &tuning_result: pimpl_->tuning_results_) {
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
        fprintf(file, "%lu;", setting.value);
      }
      fprintf(file, "\n");
    }
  }
  fclose(file);
}

// Set the flag to suppress output to true. Note that this cannot be undone.
void Tuner::SuppressOutput() {
  pimpl_->suppress_output_ = true;
}

// End of the the public tuner API
// =================================================================================================
// =================================================================================================
// Start of the private implementation

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

