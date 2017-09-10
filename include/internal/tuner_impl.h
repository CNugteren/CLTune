
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the non-publicly visible part of the tuner. It contains the header file for
// the TunerImpl class, the implemenation in the Pimpl idiom. This class contains a vector of
// KernelInfo objects, holding the actual kernels and parameters. This class interfaces between
// them. This class is also responsible for the actual tuning and the collection and dissemination
// of the results.
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

#ifndef CLTUNE_TUNER_IMPL_H_
#define CLTUNE_TUNER_IMPL_H_

// Uses either the OpenCL or CUDA back-end (CLCudaAPI C++11 headers)
#if USE_OPENCL
  #include "internal/clpp11.h"
#else
  #include "internal/cupp11.h"
#endif

#include "internal/kernel_info.h"
#include "internal/msvc.h"

// Host data-type for half-precision floating-point (16-bit)
#include "internal/half.h"

#include <string> // std::string
#include <vector> // std::vector
#include <memory> // std::shared_ptr
#include <complex> // std::complex
#include <stdexcept> // std::runtime_error

namespace cltune {
// =================================================================================================

// Shorthands for complex data-types
using float2 = std::complex<float>; // cl_float2;
using double2 = std::complex<double>; // cl_double2;

// Raw device buffer
#if USE_OPENCL
  using BufferRaw = cl_mem;
#else
  using BufferRaw = CUdeviceptr;
#endif

// Enumeration of currently supported data-types by this class
enum class MemType { kShort, kInt, kSizeT, kHalf, kFloat, kDouble, kFloat2, kDouble2 };

// See comment at top of file for a description of the class
class TunerImpl {
 // Note that everything here is public because of the Pimpl-idiom
 public:

  // Parameters
  static const double kMaxL2Norm; // This is the threshold for 'correctness'

  // Messages printed to stdout (in colours)
  static const std::string kMessageFull;
  static const std::string kMessageHead;
  static const std::string kMessageRun;
  static const std::string kMessageInfo;
  static const std::string kMessageVerbose;
  static const std::string kMessageOK;
  static const std::string kMessageWarning;
  static const std::string kMessageFailure;
  static const std::string kMessageResult;
  static const std::string kMessageBest;

  // Helper structure to store a device memory argument for a kernel
  struct MemArgument {
    size_t index;       // The kernel-argument index
    size_t size;        // The number of elements (not bytes)
    MemType type;       // The data-type (e.g. float)
    BufferRaw buffer;   // The buffer on the device
  };

  // Helper structure to hold the results of a tuning run
  struct TunerResult {
    std::string kernel_name;
    float time;
    size_t threads;
    bool status;
    KernelInfo::Configuration configuration;
  };

  // Initialize either with platform 0 and device 0 or with a custom platform/device
  explicit TunerImpl(const size_t platform_id = 0, const size_t device_id = 0);
  ~TunerImpl();

  // Starts the tuning process. This function is called directly from the Tuner API.
  void Tune();

  // Compiles and runs a kernel and returns the elapsed time
  TunerResult RunKernel(const std::string &source, const KernelInfo &kernel,
                        const size_t configuration_id, const size_t num_configurations);

  // Copies an output buffer
  template <typename T> MemArgument CopyOutputBuffer(MemArgument &argument);

  // Stores the output of the reference run into the host memory
  void StoreReferenceOutput();
  template <typename T> void DownloadReference(MemArgument &device_buffer);

  // Downloads the output of a tuning run and compares it against the reference run
  bool VerifyOutput();
  template <typename T> bool DownloadAndCompare(MemArgument &device_buffer, const size_t i);
  template <typename T> double AbsoluteDifference(const T reference, const T result);

  // Trains and uses a machine learning model based on the search space explored so far
  void ModelPrediction(const Model model_type, const float validation_fraction,
                       const size_t test_top_x_configurations);

  // Prints results of a particular kernel run
  void PrintResult(FILE* fp, const TunerResult &result, const std::string &message) const;

  // Retrieves the best tuning result
  TunerResult GetBestResult() const;

  // Loads a file from disk into a string
  std::string LoadFile(const std::string &filename);

  // Prints a header of a new section in the tuning process
  void PrintHeader(const std::string &header_name) const;

  // Specific implementations of the helper structure to get the memory-type based on a template
  // argument. Supports all enumerations of MemType.
  template <typename T> MemType GetType();

  // Rounding functions performing ceiling and division operations
  size_t CeilDiv(const size_t x, const size_t y) {
    return 1 + ((x - 1) / y);
  }
  size_t Ceil(const size_t x, const size_t y) {
    return CeilDiv(x,y)*y;
  }

  // Accessors to device data-types
  const Platform platform() const { return platform_; }
  const Device device() const { return device_; }
  const Context context() const { return context_; }
  Queue queue() const { return queue_; }

  // Device variables
  Platform platform_;
  Device device_;
  Context context_;
  Queue queue_;

  // Settings
  size_t num_runs_; // This is used for more-accurate execution time measurement
  bool has_reference_;
  bool suppress_output_;
  bool output_search_process_;
  std::string search_log_filename_;

  // The search method and its arguments
  SearchMethod search_method_;
  std::vector<double> search_args_;

  // Storage of kernel sources, arguments, and parameters
  size_t argument_counter_;
  std::vector<KernelInfo> kernels_;
  std::vector<MemArgument> arguments_input_;
  std::vector<MemArgument> arguments_output_; // these remain constant
  std::vector<MemArgument> arguments_output_copy_; // these may be modified by the kernel
  std::vector<std::pair<size_t,int>> arguments_int_;
  std::vector<std::pair<size_t,size_t>> arguments_size_t_;
  std::vector<std::pair<size_t,float>> arguments_float_;
  std::vector<std::pair<size_t,double>> arguments_double_;
  std::vector<std::pair<size_t,float2>> arguments_float2_;
  std::vector<std::pair<size_t,double2>> arguments_double2_;

  // Storage for the reference kernel and output
  std::unique_ptr<KernelInfo> reference_kernel_;
  std::vector<void*> reference_outputs_;

  // List of tuning results
  std::vector<TunerResult> tuning_results_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_TUNER_IMPL_H_
#endif
