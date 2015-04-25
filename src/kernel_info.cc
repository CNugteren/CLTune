
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the KernelInfo class (see the header for information about the class).
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

#include "cltune/kernel_info.h"

#include <cassert>

namespace cltune {
// =================================================================================================

// Initializes the name and OpenCL source-code, creates empty containers for all other member
// variables.
KernelInfo::KernelInfo(const std::string name, const std::string source,
                      std::shared_ptr<OpenCL> opencl) :
  name_(name),
  source_(source),
  parameters_(),
  configurations_(),
  constraints_(),
  opencl_(opencl),
  global_base_(), local_base_(),
  global_(), local_(),
  thread_size_modifiers_() {
}

// =================================================================================================

// Pushes a new parameter to the list of parameters
void KernelInfo::AddParameter(const std::string name, const std::vector<int> values) {
  Parameter parameter = {name, values};
  parameters_.push_back(parameter);
}

// Loops over all parameters and checks whether the given parameter name is present
bool KernelInfo::ParameterExists(const std::string parameter_name) {
  for (auto &parameter: parameters_) {
    if (parameter.name == parameter_name) { return true; }
  }
  return false;
}

// =================================================================================================

// Pushes a new item onto the list of modifiers of a particular type
void KernelInfo::AddModifier(const StringRange range, const ThreadSizeModifierType type) {
  ThreadSizeModifier modifier = {range, type};
  thread_size_modifiers_.push_back(modifier);
}

// Adds a constraint to the list of constraints
void KernelInfo::AddConstraint(ConstraintFunction valid_if,
                               const std::vector<std::string> &parameters) {
  constraints_.push_back({valid_if, parameters});
}

// =================================================================================================

// Iterates over all modifiers (e.g. add a local multiplier) and applies these values to the
// global/local thread-sizes. Modified results are kept in temporary values, but are finally
// copied back to the member variables global_ and local_.
void KernelInfo::ComputeRanges(const Configuration &config) {

  // Initializes the result vectors
  size_t num_dimensions = global_base_.dimensions();
  if (num_dimensions != local_base_.dimensions()) {
    throw Exception("Mismatching number of global/local dimensions");
  }
  std::vector<size_t> global_values(num_dimensions);
  std::vector<size_t> local_values(num_dimensions);

  // Iterates over the three dimensions (x,y,z)
  for (size_t dim=0; dim<num_dimensions; ++dim) {
    global_values[dim] = global_base_[dim];
    local_values[dim] = local_base_[dim];

    // Iterates over all the applied modifiers
    for (auto &modifier: thread_size_modifiers_) {
      std::string modifier_string = modifier.value.sizes(dim);

      // Replaces the parameter-string with the corresponding integer and processes it
      bool found_string = false;
      for (auto &setting: config) {
        if (modifier_string == setting.name) {
          switch (modifier.type) {
            case ThreadSizeModifierType::kGlobalMul: global_values[dim] *= setting.value; break;
            case ThreadSizeModifierType::kGlobalDiv: global_values[dim] /= setting.value; break;
            case ThreadSizeModifierType::kLocalMul: local_values[dim] *= setting.value; break;
            case ThreadSizeModifierType::kLocalDiv: local_values[dim] /= setting.value; break;
            default: assert(0 && "Invalid modifier type");
          }
          found_string = true;
        }
      }

      // No replacement was found, there might be something wrong with the string
      if (!found_string && modifier_string != "") {
        throw Exception("Invalid modifier: "+modifier_string);
      }
    }
  }

  // Stores the final integer results
  switch (num_dimensions) {
    case 0:
      global_ = cl::NDRange();
      local_ = cl::NDRange();
      break;
    case 1:
      global_ = cl::NDRange(global_values[0]);
      local_ = cl::NDRange(local_values[0]);
      break;
    case 2:
      global_ = cl::NDRange(global_values[0], global_values[1]);
      local_ = cl::NDRange(local_values[0], local_values[1]);
      break;
    case 3:
      global_ = cl::NDRange(global_values[0], global_values[1], global_values[2]);
      local_ = cl::NDRange(local_values[0], local_values[1], local_values[2]);
      break;
    default: assert(0 && "Invalid number of dimensions");
  }
}

// =================================================================================================

// Initializes an empty configuration (vector of name/value pairs) and kicks-off the recursive
// function to find all configurations. It also applies the user-defined constraints within.
void KernelInfo::SetConfigurations() {
  Configuration config;
  PopulateConfigurations(0, config);
}

// Iterates recursively over all permutations of the user-defined parameters. This code creates
// multiple chains, in which each chain selects a unique combination of values for all parameters.
// At the end of each chain (when all parameters are considered), the function stores the result
// into the configuration list.
void KernelInfo::PopulateConfigurations(const size_t index, const Configuration &config) {

  // End of the chain: all parameters are considered, store the resulting configuration if it is a
  // valid one according to the constraints
  if (index == parameters_.size()) {
    if (ValidConfiguration(config)) {
      configurations_.push_back(config);
    }
    return;
  }

  // This loop iterates over all values of the current parameter and calls this function
  // recursively
  Parameter parameter = parameters_[index];
  for (auto &value: parameter.values) {
    Configuration config_copy = config;
    config_copy.push_back({parameter.name, value});
    PopulateConfigurations(index+1, config_copy);
  }
}

// Loops over all user-defined constraints to check whether or not the configuration is valid.
// Assumes initially all configurations are valid, then returns false if one of the constraints has
// not been met. Constraints consist of a user-defined function and a list of parameter names, which
// are replaced by parameter values in this function.
bool KernelInfo::ValidConfiguration(const Configuration &config) {

  // Iterates over all constraints
  for (auto &constraint: constraints_) {

    // Finds the values of the parameters
    std::vector<int> values(0);
    for (auto &parameter: constraint.parameters) {
      for (auto &setting: config) {
        if (setting.name == parameter) {
          values.push_back(setting.value);
          break;
        }
      }
    }
    if (constraint.parameters.size() != values.size()) {
      throw Exception("Invalid tuning parameter constraint");
    }

    // Checks this constraint for these values
    if (!constraint.valid_if(values)) {
      return false;
    }
  }

  // Computes the global and local worksizes
  ComputeRanges(config);

  // Verifies the global/local thread-sizes against device properties
  if (!opencl_->ValidThreadSizes(global_, local_)) { return false; };

  // Everything was OK: this configuration is valid
  return true;
}

// =================================================================================================
} // namespace cltune
