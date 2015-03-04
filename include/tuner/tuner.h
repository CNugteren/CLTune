
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the externally visible Tuner class. This class contains a vector of KernelInfo
// objects, holding the actual kernels and parameters. This class interfaces between them. This
// class is also responsible for the actual tuning and the collection and dissemination of the
// results.
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

#ifndef CLTUNE_TUNER_TUNER_H_
#define CLTUNE_TUNER_TUNER_H_

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <functional>

// Include other classes
#include "tuner/internal/memory.h"
#include "tuner/internal/opencl.h"
#include "tuner/internal/kernel_info.h"
#include "tuner/internal/string_range.h"
#include "tuner/internal/searchers/full_search.h"
#include "tuner/internal/searchers/random_search.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class Tuner {
 public:
  static constexpr auto kMaxL2Norm = 1e-4; // This is the threshold for 'correctness'
  static constexpr auto kNumRuns = 1; // This is used for more-accurate execution time measurement

  // Messages printed to stdout (in colours)
  static const std::string kMessageFull;
  static const std::string kMessageHead;
  static const std::string kMessageRun;
  static const std::string kMessageInfo;
  static const std::string kMessageOK;
  static const std::string kMessageWarning;
  static const std::string kMessageFailure;
  static const std::string kMessageResult;
  static const std::string kMessageBest;

  // Helper structure to store an OpenCL memory argument for a kernel
  struct MemArgument {
    int index;          // The OpenCL kernel-argument index
    size_t size;        // The number of elements (not bytes)
    MemType type;       // The data-type (e.g. float)
    cl::Buffer buffer;  // The host memory and OpenCL buffer on the device
  };

  // Helper structure to hold the results of a tuning run
  struct TunerResult {
    std::string kernel_name;
    double time;
    size_t threads;
    bool status;
    KernelInfo::Configuration configuration;
  };

  // Exception of the tuner itself
  class Exception : public std::runtime_error {
   public:
    Exception(const std::string &message)
                   : std::runtime_error(message) { };
  };

  // Initialize either with platform 0 and device 0 or with a custom platform/device
  explicit Tuner();
  explicit Tuner(int platform_id, int device_id);
  ~Tuner();

  // Adds a new kernel to the list of tuning-kernels and returns a unique ID (to be used when
  // adding tuning parameters)
  int AddKernel(const std::string &filename, const std::string &kernel_name,
                const cl::NDRange &global, const cl::NDRange &local);

  // Sets the reference kernel. Same as the AddKernel function, but in this case there is only one
  // reference kernel. Calling this function again will overwrite the previous reference kernel.
  void SetReference(const std::string &filename, const std::string &kernel_name,
                    const cl::NDRange &global, const cl::NDRange &local);

  // Adds a new tuning parameter for a kernel with a specific ID. The parameter has a name, the
  // number of values, and a list of values.
  // TODO: Remove all following functions (those that take "const size_t id" as first argument) and
  // make the KernelInfo class publicly accessible instead.
  void AddParameter(const size_t id, const std::string parameter_name,
                    const std::initializer_list<int> values);

  // Modifies the global or local thread-size (in NDRange form) by one of the parameters (in
  // StringRange form). The modifier can be multiplication or division.
  void MulGlobalSize(const size_t id, const StringRange range);
  void DivGlobalSize(const size_t id, const StringRange range);
  void MulLocalSize(const size_t id, const StringRange range);
  void DivLocalSize(const size_t id, const StringRange range);

  // Adds a new constraint to the set of parameters (e.g. must be equal or larger than). The
  // constraints come in the form of a function object which takes a number of tuning parameters,
  // given as a vector of strings (parameter names). Their names are later substituted by actual
  // values.
  void AddConstraint(const size_t id, KernelInfo::ConstraintFunction valid_if,
                     const std::vector<std::string> &parameters);

  // Functions to add kernel-arguments for input buffers, output buffers, and scalars. Make sure to
  // call these in the order in which the arguments appear in the OpenCL kernel.
  template <typename T> void AddArgumentInput(std::vector<T> &source);
  template <typename T> void AddArgumentOutput(std::vector<T> &source);
  template <typename T> void AddArgumentScalar(const T argument);

  // Starts the tuning process: compile all kernels and run them for each permutation of the tuning-
  // parameters. Note that this might take a while.
  void Tune();

  // Prints the results of the tuning either to screen (stdout) or to a specific output-file.
  // Returns the execution time in miliseconds.
  double PrintToScreen() const;
  void PrintToFile(const std::string &filename) const;

  // Disable all further printing to stdout
  void SuppressOutput();

 private:
  // Compiles and runs a kernel and returns the elapsed time
  TunerResult RunKernel(const std::string &source, const KernelInfo &kernel,
                        const int configuration_id, const int num_configurations);

  // Sets an OpenCL buffer to zero
  template <typename T> void ResetMemArgument(MemArgument &argument);

  // Stores the output of the reference run into the host memory
  void StoreReferenceOutput();
  template <typename T> void DownloadReference(const MemArgument &device_buffer);

  // Downloads the output of a tuning run and compares it against the reference run
  bool VerifyOutput();
  template <typename T> bool DownloadAndCompare(const MemArgument &device_buffer, const size_t i);

  // Prints results of a particular kernel run
  void PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const;

  // Loads a file from disk into a string
  std::string LoadFile(const std::string &filename);

  // Prints a header of a new section in the tuning process
  void PrintHeader(const std::string &header_name) const;

  // OpenCL platform
  std::shared_ptr<OpenCL> opencl_;

  // Settings
  bool has_reference_;
  bool suppress_output_;

  // Storage of kernel sources, arguments, and parameters
  int argument_counter_;
  std::vector<KernelInfo> kernels_;
  std::vector<MemArgument> arguments_input_;
  std::vector<MemArgument> arguments_output_;
  std::vector<std::pair<int,int>> arguments_scalar_;

  // Storage for the reference kernel and output
  std::shared_ptr<KernelInfo> reference_kernel_;
  std::vector<void*> reference_outputs_;

  // List of tuning results
  std::vector<TunerResult> tuning_results_;

};

// =================================================================================================
} // namespace cltune

// CLTUNE_TUNER_TUNER_H_
#endif
