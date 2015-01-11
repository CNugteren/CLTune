
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// The CLTune project follows the Google C++ styleguide and uses a tab-size of two spaces and a
// max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the KernelInfo class (see the header for information about the class).
//
// =================================================================================================

#include "tuner/internal/kernel_info.h"

#include <cassert>

namespace cltune {
// =================================================================================================

// Initializes the name and OpenCL source-code, creates empty containers for all other member
// variables.
KernelInfo::KernelInfo(std::string name, std::string source) :
  name_(name),
  source_(source),
  parameters_(),
  configuration_list_(),
  constraints_(),
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
void KernelInfo::AddConstraint(const std::string parameter_1, const ConstraintType type,
                               const std::string parameter_2) {
  Constraint constraint = {parameter_1, type, parameter_2, kNoOp, "", kNoOp, ""};
  constraints_.push_back(constraint);
}

// Same as above, but now with an extra operation on the second parameter
void KernelInfo::AddConstraint(const std::string parameter_1, const ConstraintType type,
                               const std::string parameter_2, const OperatorType op,
                               const std::string parameter_3) {
  Constraint constraint = {parameter_1, type, parameter_2, op, parameter_3, kNoOp, ""};
  constraints_.push_back(constraint);
}

// Same as above, but now with two extra operations on the second parameter
void KernelInfo::AddConstraint(const std::string parameter_1, const ConstraintType type,
                               const std::string parameter_2, const OperatorType op_1,
                               const std::string parameter_3, const OperatorType op_2,
                               const std::string parameter_4) {
  Constraint constraint = {parameter_1, type, parameter_2, op_1, parameter_3, op_2, parameter_4};
  constraints_.push_back(constraint);
}

// =================================================================================================

// Iterates over all modifiers (e.g. add a local multiplier) and applies these values to the
// global/local thread-sizes. Modified results are kept in temporary values, but are finally
// copied back to the member variables global_ and local_.
void KernelInfo::ComputeRanges(const std::vector<Configuration> &configuration) {

  // Initializes the result vectors
  size_t num_dimensions = global_base_.dimensions();
  if (num_dimensions != local_base_.dimensions()) {
    throw KernelInfoException("Mismatching number of global/local dimensions");
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
      for (auto &parameter: configuration) {
        if (modifier_string == parameter.name) {
          switch (modifier.type) {
            case kGlobalMul: global_values[dim] *= parameter.value; break;
            case kGlobalDiv: global_values[dim] /= parameter.value; break;
            case kLocalMul: local_values[dim] *= parameter.value; break;
            case kLocalDiv: local_values[dim] /= parameter.value; break;
            default: assert(0 && "Invalid modifier type");
          }
          found_string = true;
        }
      }

      // No replacement was found, there might be something wrong with the string
      if (!found_string && modifier_string != "") {
        throw KernelInfoException("Invalid modifier: "+modifier_string);
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

// Initializes an empty configuration (name/value pair) and kicks-off the recursive function to
// find the configuration list. It also applies the user-defined constraints within.
void KernelInfo::SetConfigurationList() {
  std::vector<Configuration> configuration;
  PopulateConfigurations(0, configuration);
}

// Iterates recursively over all permutations of the user-defined parameters. This code creates
// multiple chains, in which each chain selects a unique combination of values for all parameters.
// At the end of each chain (when all parameters are considered), the function stores the result
// into the configuration list.
void KernelInfo::PopulateConfigurations(const size_t index,
                                        const std::vector<Configuration> &configuration) {

  // End of the chain: all parameters are considered, store the resulting configuration if it is a
  // valid one according to the constraints
  if (index == parameters_.size()) {
    if (ValidConfiguration(configuration)) {
      configuration_list_.push_back(configuration);
    }
    return;
  }

  // This loop iterates over all values of the current parameter and calls this function
  // recursively
  Parameter parameter = parameters_[index];
  for (auto &value: parameter.values) {
    std::vector<Configuration> configuration_copy = configuration;
    configuration_copy.push_back({parameter.name, value});
    PopulateConfigurations(index+1, configuration_copy);
  }
}

// Loops over all user-defined constraints to check whether or not the configuration is valid.
// Assumes initially all configurations are valid, then returns flag if one of the constraints has
// not been met. Constraints operate on two parameters and are of a certain type, see the
// ConstraintType enumeration for all supported types.
// TODO: Make this function more generic
bool KernelInfo::ValidConfiguration(const std::vector<Configuration> &configuration) {
  for (auto &constraint: constraints_) {

    // Finds the combination of the two parameters (if it exists at all)
    for (auto &p1: configuration) {
      if (p1.name == constraint.parameter_1) {
        for (auto &p2: configuration) {
          if (p2.name == constraint.parameter_2) {
            int value_1 = p1.value;
            int value_2 = p2.value;

            // Calculates the second value in case of a third parameter is present
            if (constraint.op_1 != kNoOp) {
              for (auto &p3: configuration) {
                if (p3.name == constraint.parameter_3) {
                  switch (constraint.op_1) {
                    case kMultipliedBy: value_2 *= p3.value; break;
                    case kDividedBy: value_2 /= p3.value; break;
                    default: throw KernelInfoException("Invalid operation type");
                  }

                  // Calculates the second value in case of a fourth parameter is present
                  if (constraint.op_2 != kNoOp) {
                    for (auto &p4: configuration) {
                      if (p4.name == constraint.parameter_4) {
                        switch (constraint.op_2) {
                          case kMultipliedBy: value_2 *= p4.value; break;
                          case kDividedBy: value_2 /= p4.value; break;
                          default: throw KernelInfoException("Invalid operation type");
                        }
                        break;
                      }
                    }
                  }
                  break;
                }
              }
            }

            // Performs the constraint check
            switch (constraint.type) {
              case kEqual: if (!(value_1 == value_2)) { return false; } break;
              case kLargerThan: if (!(value_1 > value_2)) { return false; } break;
              case kLargerEqual: if (!(value_1 >= value_2)) { return false; } break;
              case kSmallerThan: if (!(value_1 < value_2)) { return false; } break;
              case kSmallerEqual: if (!(value_1 <= value_2)) { return false; } break;
              case kMultipleOf:
                if (!((value_1/value_2)*value_2 == value_1)) { return false; }
                break;
              default: throw KernelInfoException("Invalid constraint type");
            }
          }
        }
      }
    }
  }
  return true;
}

// =================================================================================================
} // namespace cltune
