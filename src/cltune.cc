
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
#include "cltune/tuner_impl.h"

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

// =================================================================================================
} // namespace cltune
