
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// The CLTune project follows the Google C++ styleguide and uses a tab-size of two spaces and a
// max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the KernelInfo class which holds information for a single kernel including
// all its parameters and settings. It holds the kernel name and source-code as a string, it holds
// the global and local NDRange settings, and it holds the parameters set by the user (and its
// permutations).
//
// =================================================================================================

#ifndef CLBLAS_TUNER_KERNEL_INFO_H_
#define CLBLAS_TUNER_KERNEL_INFO_H_

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

// The C++ OpenCL wrapper
#include "cl.hpp"

// Include other classes and structures
#include "tuner/internal/string_range.h"

namespace cltune {
// =================================================================================================

// Enumeration of modifiers to global/local thread-sizes
enum ThreadSizeModifierType { kGlobalMul, kGlobalDiv, kLocalMul, kLocalDiv };

// Enumeration of equalities/inequalities on parameter
enum ConstraintType { kEqual, kLargerThan, kLargerEqual, kSmallerThan, kSmallerEqual, kMultipleOf };

// Enumeration of operations on parameter
enum OperatorType { kNoOp, kMultipliedBy, kDividedBy };

// =================================================================================================

// See comment at top of file for a description of the class
class KernelInfo {
 public:

  // Helper structure holding a parameter name and a list of all values
  struct Parameter {
    std::string name;
    std::vector<int> values;
  };

  // Helper structure holding a configuration: a name and a value 
  struct Configuration {
    std::string name;
    int value;
    std::string GetDefine() const { return "#define "+name+" "+GetValueString()+"\n"; }
    std::string GetConfig() const { return name+" "+GetValueString(); }
    std::string GetValueString() const { return std::to_string((long long)value); }
  };

  // Helper structure holding a modifier: its value and its type
  struct ThreadSizeModifier {
    StringRange value;
    ThreadSizeModifierType type;
  };

  // Helper structure holding a constraint on parameters
  // TODO: Make this more generic with a vector of parameters and operators
  struct Constraint {
    std::string parameter_1;
    ConstraintType type;
    std::string parameter_2;
    OperatorType op_1;
    std::string parameter_3;
    OperatorType op_2;
    std::string parameter_4;
  };

  // Temporary structure
  struct SupportKernel {
    std::string name;
    cl::NDRange global;
    cl::NDRange local;
  };

  // Exception of the KernelInfo class
  class KernelInfoException : public std::runtime_error {
   public:
    KernelInfoException(const std::string &message)
                        : std::runtime_error(message) { };
  };

  // Initializes the class with a given name and a string of OpenCL source-code
  explicit KernelInfo(std::string name, std::string source);

  // Accessors (getters)
  std::string name() const { return name_; }
  std::string source() const { return source_; }
  std::vector<Parameter> parameters() const { return parameters_; }
  cl::NDRange global_base() const { return global_base_; }
  cl::NDRange local_base() const { return local_base_; }
  cl::NDRange global() const { return global_; }
  cl::NDRange local() const { return local_; }
  std::vector<std::vector<Configuration>> configuration_list() { return configuration_list_; }

  // Accessors (setters) - Note that these also pre-set the final global/local size
  void set_global_base(cl::NDRange global) { global_base_ = global; global_ = global; }
  void set_local_base(cl::NDRange local) { local_base_ = local; local_ = local; }

  // Adds a new parameter with a name and a vector of possible values
  void AddParameter(const std::string name, const std::vector<int> values);

  // Checks wheter a parameter exists, returns "true" if it does exist
  bool ParameterExists(const std::string parameter_name);

  // Specifies a modifier in the form of a StringRange to the global/local thread-sizes. This
  // modifier has to contain (per-dimension) the name of a single parameter or an empty string. The
  // supported modifiers are given by the ThreadSizeModifierType enumeration.
  void AddModifier(const StringRange range, const ThreadSizeModifierType type);

  // Adds a new constraint to the set of parameters (e.g. must be equal or larger than)
  // TODO: Combine the below three functions and make them more generic.
  void AddConstraint(const std::string parameter_1, const ConstraintType type,
                     const std::string parameter_2);

  // Also adds a constraint, but the second parameter is now modified by an operation "op" with
  // respect to a third parameter (e.g. multiplication)
  void AddConstraint(const std::string parameter_1, const ConstraintType type,
                     const std::string parameter_2, const OperatorType op,
                     const std::string parameter_3);

  // As above, but with a second operation and a fourth parameter
  void AddConstraint(const std::string parameter_1, const ConstraintType type,
                     const std::string parameter_2, const OperatorType op_1,
                     const std::string parameter_3, const OperatorType op_2,
                     const std::string parameter_4);

  // Computes the global/local ranges (in NDRange-form) based on all global/local thread-sizes (in
  // StringRange-form) and a single permutation (i.e. a configuration) containing a list of all
  // parameter names and their current values.
  void ComputeRanges(const std::vector<Configuration> &configuration);

  // Computes all permutations based on the parameters and their values (the configuration list).
  // The result is stored as a member variable.
  void SetConfigurationList();
  
 private:
  // Called recursively internally by SetConfigurationList 
  void PopulateConfigurations(const size_t index, const std::vector<Configuration> &configuration);

  // Returns whether or not a given configuration is valid. This check is based on the user-supplied
  // constraints.
  bool ValidConfiguration(const std::vector<Configuration> &configuration);

  // Member variables
  std::string name_;
  std::string source_;
  std::vector<Parameter> parameters_;
  std::vector<std::vector<Configuration>> configuration_list_;
  std::vector<Constraint> constraints_;

  // Global/local thread-sizes
  cl::NDRange global_base_;
  cl::NDRange local_base_;
  cl::NDRange global_;
  cl::NDRange local_;

  // Multipliers and dividers for global/local thread-sizes
  std::vector<ThreadSizeModifier> thread_size_modifiers_;
};

// =================================================================================================
} // namespace cltune

// CLBLAS_TUNER_KERNEL_INFO_H_
#endif
