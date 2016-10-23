
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the KernelInfo class which holds information for a single kernel including
// all its parameters and settings. It holds the kernel name and source-code as a string, it holds
// the global and local NDRange settings, and it holds the parameters set by the user (and its
// permutations).
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

#ifndef CLTUNE_KERNEL_INFO_H_
#define CLTUNE_KERNEL_INFO_H_

#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>

// Uses either the OpenCL or CUDA back-end (CLCudaAPI C++11 headers)
#if USE_OPENCL
  #include "internal/clpp11.h"
#else
  #include "internal/cupp11.h"
#endif

#include "cltune.h"
#include "internal/msvc.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class KernelInfo {
 public:

  // Enumeration of modifiers to global/local thread-sizes
  enum class ThreadSizeModifierType { kGlobalMul, kGlobalDiv, kLocalMul, kLocalDiv };

  // Helper structure holding a parameter name and a list of all values
  struct Parameter {
    std::string name;
    std::vector<size_t> values;
  };

  // Helper structure holding a setting: a name and a value. Multiple settings combined make a
  // single configuration.
  struct Setting {
    std::string name;
    size_t value;
    std::string GetDefine() const { return "#define "+name+" "+GetValueString()+"\n"; }
    std::string GetConfig() const { return name+" "+GetValueString(); }
    std::string GetDatabase() const { return "{\""+name+"\","+GetValueString()+"}"; }
    std::string GetValueString() const { return std::to_string(static_cast<long long>(value)); }
  };
  using Configuration = std::vector<Setting>;

  // Helper structure holding a modifier: its value and its type
  struct ThreadSizeModifier {
    StringRange value;
    ThreadSizeModifierType type;
  };

  // Helper structure holding a constraint on parameters. This constraint consists of a constraint
  // function object and a vector of paramater names represented as strings.
  struct Constraint {
    ConstraintFunction valid_if;
    std::vector<std::string> parameters;
  };

  // As above, but for local memory size.
  struct LocalMemory {
    LocalMemoryFunction amount;
    std::vector<std::string> parameters;
  };

  // Exception of the KernelInfo class
  class Exception : public std::runtime_error {
   public:
    Exception(const std::string &message): std::runtime_error(message) { }
  };

  // Initializes the class with a given name and a string of kernel source-code
  explicit PUBLIC_API KernelInfo(const std::string name, const std::string source, const Device &device);

  // Accessors (getters)
  std::string name() const { return name_; }
  std::string source() const { return source_; }
  std::vector<Parameter> parameters() const { return parameters_; }
  IntRange global_base() const { return global_base_; }
  IntRange local_base() const { return local_base_; }
  IntRange global() const { return global_; }
  IntRange local() const { return local_; }
  std::vector<Configuration> configurations() { return configurations_; }

  // Accessors (setters) - Note that these also pre-set the final global/local size
  void set_global_base(IntRange global) { global_base_ = global; global_ = global; }
  void set_local_base(IntRange local) { local_base_ = local; local_ = local; }

  // Prepend to the source-code
  void PUBLIC_API PrependSource(const std::string &extra_source);

  // Adds a new parameter with a name and a vector of possible values
  void PUBLIC_API AddParameter(const std::string &name, const std::vector<size_t> &values);

  // Checks wheter a parameter exists, returns "true" if it does exist
  bool PUBLIC_API ParameterExists(const std::string parameter_name);

  // Specifies a modifier in the form of a StringRange to the global/local thread-sizes. This
  // modifier has to contain (per-dimension) the name of a single parameter or an empty string. The
  // supported modifiers are given by the ThreadSizeModifierType enumeration.
  void PUBLIC_API AddModifier(const StringRange range, const ThreadSizeModifierType type);

  // Adds a new constraint to the set of parameters (e.g. must be equal or larger than). The
  // constraints come in the form of a function object which takes a number of tuning parameters,
  // given as a vector of strings (parameter names). Their names are later substituted by actual
  // values.
  void PUBLIC_API AddConstraint(ConstraintFunction valid_if, const std::vector<std::string> &parameters);

  // As above, but for local memory usage
  void PUBLIC_API SetLocalMemoryUsage(LocalMemoryFunction amount, const std::vector<std::string> &parameters);

  // Computes the global/local ranges (in NDRange-form) based on all global/local thread-sizes (in
  // StringRange-form) and a single permutation (i.e. a configuration) containing a list of all
  // parameter names and their current values.
  void PUBLIC_API ComputeRanges(const Configuration &config);

  // Computes all permutations based on the parameters and their values (the configuration list).
  // The result is stored as a member variable.
  void PUBLIC_API SetConfigurations();
  
 private:
  // Called recursively internally by SetConfigurations 
  void PopulateConfigurations(const size_t index, const Configuration &config);

  // Returns whether or not a given configuration is valid. This check is based on the user-supplied
  // constraints.
  bool ValidConfiguration(const Configuration &config);

  // Member variables
  std::string name_;
  std::string source_;
  std::vector<Parameter> parameters_;
  std::vector<Configuration> configurations_;
  std::vector<Constraint> constraints_;
  LocalMemory local_memory_;

  Device device_;

  // Global/local thread-sizes
  IntRange global_base_;
  IntRange local_base_;
  IntRange global_;
  IntRange local_;

  // Multipliers and dividers for global/local thread-sizes
  std::vector<ThreadSizeModifier> thread_size_modifiers_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_KERNEL_INFO_H_
#endif
