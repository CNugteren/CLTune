
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

// The corresponding header file
#include "cltune.h"

// And the implemenation (Pimpl idiom)
#include "internal/tuner_impl.h"

#include <iostream> // FILE
#include <limits> // std::numeric_limits

namespace cltune {
// =================================================================================================

// The implemenation of the constructors and destructors are hidden in the TunerImpl class
Tuner::Tuner():
    pimpl(new TunerImpl()) {
}
Tuner::Tuner(size_t platform_id, size_t device_id):
    pimpl(new TunerImpl(platform_id, device_id)) {
}
Tuner::~Tuner() {
}

// =================================================================================================

// Loads the OpenCL source-code from a file and calls the function-overload below.
size_t Tuner::AddKernel(const std::vector<std::string> &filenames, const std::string &kernel_name,
                        const IntRange &global, const IntRange &local) {
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl->LoadFile(filename);
  }
  return AddKernelFromString(source, kernel_name, global, local);
}

// Loads the OpenCL source-code from a string and creates a new variable of type KernelInfo to store
// all the kernel-information.
size_t Tuner::AddKernelFromString(const std::string &source, const std::string &kernel_name,
                                  const IntRange &global, const IntRange &local) {
  pimpl->kernels_.push_back(KernelInfo(kernel_name, source, pimpl->device()));
  auto id = pimpl->kernels_.size() - 1;
  pimpl->kernels_[id].set_global_base(global);
  pimpl->kernels_[id].set_local_base(local);
  return id;
}

// =================================================================================================

// Sets the reference kernel (source-code location, kernel name, global/local thread-sizes) and
// sets a flag to indicate that there is now a reference. Calling this function again will simply
// overwrite the old reference.
void Tuner::SetReference(const std::vector<std::string> &filenames, const std::string &kernel_name,
                         const IntRange &global, const IntRange &local) {
  auto source = std::string{};
  for (auto &filename: filenames) {
    source += pimpl->LoadFile(filename);
  }
  SetReferenceFromString(source, kernel_name, global, local);
}
void Tuner::SetReferenceFromString(const std::string &source, const std::string &kernel_name,
                                   const IntRange &global, const IntRange &local) {
  pimpl->has_reference_ = true;
  pimpl->reference_kernel_.reset(new KernelInfo(kernel_name, source, pimpl->device()));
  pimpl->reference_kernel_->set_global_base(global);
  pimpl->reference_kernel_->set_local_base(local);
}

// =================================================================================================

// Adds parameters for a kernel to tune. Also checks whether this parameter already exists.
void Tuner::AddParameter(const size_t id, const std::string &parameter_name,
                         const std::initializer_list<size_t> &values) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  if (pimpl->kernels_[id].ParameterExists(parameter_name)) {
    throw std::runtime_error("Parameter already exists");
  }
  pimpl->kernels_[id].AddParameter(parameter_name, values);
}

// As above, but now adds a single valued parameter to the reference
void Tuner::AddParameterReference(const std::string &parameter_name, const size_t value) {
  auto value_string = std::string{std::to_string(static_cast<long long>(value))};
  pimpl->reference_kernel_->PrependSource("#define "+parameter_name+" "+value_string);
}

// =================================================================================================

// These functions forward their work (adding a modifier to global/local thread-sizes) to an object
// of KernelInfo class
void Tuner::MulGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalMul);
}
void Tuner::DivGlobalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kGlobalDiv);
}
void Tuner::MulLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalMul);
}
void Tuner::DivLocalSize(const size_t id, const StringRange range) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  pimpl->kernels_[id].AddModifier(range, KernelInfo::ThreadSizeModifierType::kLocalDiv);
}

// Adds a contraint to the list of constraints for a particular kernel. First checks whether the
// kernel exists and whether the parameters exist.
void Tuner::AddConstraint(const size_t id, ConstraintFunction valid_if,
                          const std::vector<std::string> &parameters) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl->kernels_[id].ParameterExists(parameter)) {
      throw std::runtime_error("Invalid parameter");
    }
  }
  pimpl->kernels_[id].AddConstraint(valid_if, parameters);
}

// As above, but for the local memory usage
void Tuner::SetLocalMemoryUsage(const size_t id, LocalMemoryFunction amount,
                                const std::vector<std::string> &parameters) {
  if (id >= pimpl->kernels_.size()) { throw std::runtime_error("Invalid kernel ID"); }
  for (auto &parameter: parameters) {
    if (!pimpl->kernels_[id].ParameterExists(parameter)) {
      throw std::runtime_error("Invalid parameter");
    }
  }
 pimpl->kernels_[id].SetLocalMemoryUsage(amount, parameters);
}


// =================================================================================================

// Creates a new buffer of type Memory (containing both host and device data) based on a source
// vector of data. Then, upload it to the device and store the argument in a list.
template <typename T>
void Tuner::AddArgumentInput(const std::vector<T> &source) {
  auto device_buffer = Buffer(pimpl->context(), source.size()*sizeof(T));
  device_buffer.Write(pimpl->queue(), source.size()*sizeof(T), source);
  auto argument = TunerImpl::MemArgument{pimpl->argument_counter_++, source.size(),
                                         pimpl->GetType<T>(), device_buffer};
  pimpl->arguments_input_.push_back(argument);
}

// Compiles the function for various data-types
template void Tuner::AddArgumentInput<int>(const std::vector<int>&);
template void Tuner::AddArgumentInput<size_t>(const std::vector<size_t>&);
template void Tuner::AddArgumentInput<float>(const std::vector<float>&);
template void Tuner::AddArgumentInput<double>(const std::vector<double>&);
template void Tuner::AddArgumentInput<float2>(const std::vector<float2>&);
template void Tuner::AddArgumentInput<double2>(const std::vector<double2>&);

// Similar to the above function, but now marked as output buffer. Output buffers are special in the
// sense that they will be checked in the verification process.
template <typename T>
void Tuner::AddArgumentOutput(const std::vector<T> &source) {
  auto device_buffer = Buffer(pimpl->context(), source.size()*sizeof(T));
  auto argument = TunerImpl::MemArgument{pimpl->argument_counter_++, source.size(),
                                         pimpl->GetType<T>(), device_buffer};
  pimpl->arguments_output_.push_back(argument);
}

// Compiles the function for various data-types
template void Tuner::AddArgumentOutput<int>(const std::vector<int>&);
template void Tuner::AddArgumentOutput<size_t>(const std::vector<size_t>&);
template void Tuner::AddArgumentOutput<float>(const std::vector<float>&);
template void Tuner::AddArgumentOutput<double>(const std::vector<double>&);
template void Tuner::AddArgumentOutput<float2>(const std::vector<float2>&);
template void Tuner::AddArgumentOutput<double2>(const std::vector<double2>&);

// Sets a scalar value as an argument to the kernel. Since a vector of scalars of any type doesn't
// exist, there is no general implemenation. Instead, each data-type has its specialised version in
// which it stores to a specific vector.
template <> void Tuner::AddArgumentScalar<int>(const int argument) {
  pimpl->arguments_int_.push_back({pimpl->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<size_t>(const size_t argument) {
  pimpl->arguments_size_t_.push_back({pimpl->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<float>(const float argument) {
  pimpl->arguments_float_.push_back({pimpl->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<double>(const double argument) {
  pimpl->arguments_double_.push_back({pimpl->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<float2>(const float2 argument) {
  pimpl->arguments_float2_.push_back({pimpl->argument_counter_++, argument});
}
template <> void Tuner::AddArgumentScalar<double2>(const double2 argument) {
  pimpl->arguments_double2_.push_back({pimpl->argument_counter_++, argument});
}

// =================================================================================================

// Use full search as a search strategy. This is the default method.
void Tuner::UseFullSearch() {
  pimpl->search_method_ = SearchMethod::FullSearch;
}

// Use random search as a search strategy.
void Tuner::UseRandomSearch(const double fraction) {
  pimpl->search_method_ = SearchMethod::RandomSearch;
  pimpl->search_args_.push_back(fraction);
}

// Use simulated annealing as a search strategy.
void Tuner::UseAnnealing(const double fraction, const double max_temperature) {
  pimpl->search_method_ = SearchMethod::Annealing;
  pimpl->search_args_.push_back(fraction);
  pimpl->search_args_.push_back(max_temperature);
}

// Use PSO as a search strategy.
void Tuner::UsePSO(const double fraction, const size_t swarm_size, const double influence_global,
                   const double influence_local, const double influence_random) {
  pimpl->search_method_ = SearchMethod::PSO;
  pimpl->search_args_.push_back(fraction);
  pimpl->search_args_.push_back(static_cast<double>(swarm_size));
  pimpl->search_args_.push_back(influence_global);
  pimpl->search_args_.push_back(influence_local);
  pimpl->search_args_.push_back(influence_random);
}


// Output the search process to a file. This is disabled per default.
void Tuner::OutputSearchLog(const std::string &filename) {
  pimpl->output_search_process_ = true;
  pimpl->search_log_filename_ = filename;
}

// =================================================================================================

// Starts the tuning process. See the TunerImpl's implemenation for details
void Tuner::Tune() {
  pimpl->Tune();
}

// =================================================================================================

// Fits a machine learning model. See the TunerImpl's implemenation for details
void Tuner::ModelPrediction(const Model model_type, const float validation_fraction,
                            const size_t test_top_x_configurations) {
  pimpl->ModelPrediction(model_type, validation_fraction, test_top_x_configurations);
}

// =================================================================================================

// Iterates over all tuning results and prints each parameter configuration and the corresponding
// timing-results. Printing is to stdout.
double Tuner::PrintToScreen() const {

  // Finds the best result
  auto best_result = pimpl->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Aborts if there was no best time found
  if (best_time == std::numeric_limits<double>::max()) {
    pimpl->PrintHeader("No tuner results found");
    return 0.0;
  }

  // Prints all valid results and the one with the lowest execution time
  pimpl->PrintHeader("Printing results to stdout");
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && tuning_result.time != std::numeric_limits<double>::max()) {
      pimpl->PrintResult(stdout, tuning_result, pimpl->kMessageResult);
    }
  }
  pimpl->PrintHeader("Printing best result to stdout");
  pimpl->PrintResult(stdout, best_result, pimpl->kMessageBest);

  // Return the best time
  return best_time;
}

// Prints the best result in a neatly formatted C++ database format to screen
void Tuner::PrintFormatted() const {

  // Finds the best result
  auto best_result = pimpl->tuning_results_[0];
  auto best_time = std::numeric_limits<double>::max();
  for (auto &tuning_result: pimpl->tuning_results_) {
    if (tuning_result.status && best_time >= tuning_result.time) {
      best_result = tuning_result;
      best_time = tuning_result.time;
    }
  }

  // Prints the best result in C++ database format
  auto count = size_t{0};
  pimpl->PrintHeader("Printing best result in database format to stdout");
  fprintf(stdout, "{ \"%s\", { ", pimpl->device().Name().c_str());
  for (auto &setting: best_result.configuration) {
    fprintf(stdout, "%s", setting.GetDatabase().c_str());
    if (count < best_result.configuration.size()-1) {
      fprintf(stdout, ", ");
    }
    count++;
  }
  fprintf(stdout, " } }\n");
}

// Outputs all results in a JSON database format
void Tuner::PrintJSON(const std::string &filename,
                      const std::vector<std::pair<std::string,std::string>> &descriptions) const {

  // Prints the best result in JSON database format
  pimpl->PrintHeader("Printing results to file in JSON format");
  auto file = fopen(filename.c_str(), "w");
  auto device_type = pimpl->device().Type();
  fprintf(file, "{\n");
  for (auto &description: descriptions) {
    fprintf(file, "  \"%s\": \"%s\",\n", description.first.c_str(), description.second.c_str());
  }
  fprintf(file, "  \"vendor\": \"%s\",\n", pimpl->device().Vendor().c_str());
  fprintf(file, "  \"type\": \"%s\",\n", device_type.c_str());
  fprintf(file, "  \"device\": \"%s\",\n", pimpl->device().Name().c_str());
  fprintf(file, "  \"results\": [\n");

  // Loops over all the results
  auto num_results = pimpl->tuning_results_.size();
  for (auto r=size_t{0}; r<num_results; ++r) {
    auto result = pimpl->tuning_results_[r];
    fprintf(file, "    {\n");
    fprintf(file, "      \"kernel\": \"%s\",\n", result.kernel_name.c_str());
    fprintf(file, "      \"time\": %.3lf,\n", result.time);

    // Loops over all the parameters for this result
    fprintf(file, "      \"parameters\": {");
    auto num_configs = result.configuration.size();
    for (auto p=size_t{0}; p<num_configs; ++p) {
      auto config = result.configuration[p];
      fprintf(file, "\"%s\": %lu", config.name.c_str(), config.value);
      if (p < num_configs-1) { fprintf(file, ","); }
    }
    fprintf(file, "}\n");

    // The footer
    fprintf(file, "    }");
    if (r < num_results-1) { fprintf(file, ","); }
    fprintf(file, "\n");
  }
  fprintf(file, "  ]\n");
  fprintf(file, "}\n");
  fclose(file);
}

// Same as PrintToScreen, but now outputs into a file and does not mark the best-case
void Tuner::PrintToFile(const std::string &filename) const {
  pimpl->PrintHeader("Printing results to file: "+filename);
  auto file = fopen(filename.c_str(), "w");
  std::vector<std::string> processed_kernels;
  for (auto &tuning_result: pimpl->tuning_results_) {
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
  pimpl->suppress_output_ = true;
}

// =================================================================================================
} // namespace cltune
