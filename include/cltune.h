
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the externally visible Tuner class. This forms the public API, implemenation
// details are hidden in the TunerImpl class.
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

#ifndef CLTUNE_CLTUNE_H_
#define CLTUNE_CLTUNE_H_

#include <string> // std::string
#include <vector> // std::vector
#include <memory> // std::unique_ptr
#include <functional> // std::function
#include <utility> // std::pair
#include <unordered_map> // std::unordered_map

// Exports library functions under Windows when building a DLL. See also:
// https://msdn.microsoft.com/en-us/library/a90k134d.aspx
#if defined(_WIN32) && defined(CLTUNE_DLL)
  #if defined(COMPILING_DLL)
    #define PUBLIC_API __declspec(dllexport)
  #else
    #define PUBLIC_API __declspec(dllimport)
  #endif
#else
  #define PUBLIC_API
#endif

namespace cltune {
// =================================================================================================

// Forward declaration of the implemenation class
class TunerImpl;

// CLTune's custom data-types
using IntRange = std::vector<size_t>;
using StringRange = std::vector<std::string>;
using ConstraintFunction = std::function<bool(std::vector<size_t>)>;
using LocalMemoryFunction = std::function<size_t(std::vector<size_t>)>;

// Enumeration for search strategies
enum class SearchMethod{FullSearch, RandomSearch, Annealing, PSO};

// Machine learning models
enum class Model { kLinearRegression, kNeuralNetwork };

// The tuner class and its public API
class Tuner {
 public:

  // Initializes the tuner either with platform 0 and device 0 or with a custom platform/device
  explicit PUBLIC_API Tuner();
  explicit PUBLIC_API Tuner(size_t platform_id, size_t device_id);
  PUBLIC_API ~Tuner();

  // Adds a new kernel to the list of tuning-kernels and returns a unique ID (to be used when
  // adding tuning parameters). Either loads the source from filenames or from string.
  size_t PUBLIC_API AddKernel(const std::vector<std::string> &filenames, const std::string &kernel_name,
                              const IntRange &global, const IntRange &local);
  size_t PUBLIC_API AddKernelFromString(const std::string &source, const std::string &kernel_name,
                                        const IntRange &global, const IntRange &local);

  // Sets the reference kernel. Same as the AddKernel function, but in this case there is only one
  // reference kernel. Calling this function again will overwrite the previous reference kernel.
  void PUBLIC_API SetReference(const std::vector<std::string> &filenames,
                               const std::string &kernel_name,
                               const IntRange &global, const IntRange &local);
  void PUBLIC_API SetReferenceFromString(const std::string &source,
                                         const std::string &kernel_name,
                                         const IntRange &global, const IntRange &local);

  // Adds a new tuning parameter for a kernel with a specific ID. The parameter has a name, the
  // number of values, and a list of values.
  void PUBLIC_API AddParameter(const size_t id, const std::string &parameter_name,
                               const std::vector<size_t> &values);

  // As above, but now adds a single valued parameter to the reference
  void PUBLIC_API AddParameterReference(const std::string &parameter_name, const size_t value);

  // Modifies the global or local thread-size (integers) by one of the parameters (strings). The
  // modifier can be multiplication or division.
  void PUBLIC_API MulGlobalSize(const size_t id, const StringRange range);
  void PUBLIC_API DivGlobalSize(const size_t id, const StringRange range);
  void PUBLIC_API MulLocalSize(const size_t id, const StringRange range);
  void PUBLIC_API DivLocalSize(const size_t id, const StringRange range);

  // Adds a new constraint to the set of parameters (e.g. must be equal or larger than). The
  // constraints come in the form of a function object which takes a number of tuning parameters,
  // given as a vector of strings (parameter names). Their names are later substituted by actual
  // values.
  void PUBLIC_API AddConstraint(const size_t id, ConstraintFunction valid_if,
                                const std::vector<std::string> &parameters);

  // As above, but for local memory usage. If this function is not called, it is assumed that the
  // local memory usage is 0: no configurations will be excluded because of too much local memory.
  void PUBLIC_API SetLocalMemoryUsage(const size_t id, LocalMemoryFunction amount,
                                      const std::vector<std::string> &parameters);

  // Functions to add kernel-arguments for input buffers, output buffers, and scalars. Make sure to
  // call these in the order in which the arguments appear in the kernel.
  template <typename T> void AddArgumentInput(const std::vector<T> &source);
  template <typename T> void AddArgumentOutput(const std::vector<T> &source);
  template <typename T> void AddArgumentScalar(const T argument);

  // Configures a specific search method. The default search method is "FullSearch". These are
  // implemented as separate functions since they each take a different number of arguments.
  void PUBLIC_API UseFullSearch();
  void PUBLIC_API UseRandomSearch(const double fraction);
  void PUBLIC_API UseAnnealing(const double fraction, const double max_temperature);
  void PUBLIC_API UsePSO(const double fraction, const size_t swarm_size, const double influence_global,
                         const double influence_local, const double influence_random);

  // Outputs the search process to a file
  void PUBLIC_API OutputSearchLog(const std::string &filename);

  // Starts the tuning process: compile all kernels and run them for each permutation of the tuning-
  // parameters. Note that this might take a while.
  void PUBLIC_API Tune();

  // Trains a machine learning model based on the search space explored so far. Then, all the
  // missing data-points are estimated based on this model. This is only useful if a fraction of
  // the search space is explored, as is the case when doing random-search.
  void PUBLIC_API ModelPrediction(const Model model_type, const float validation_fraction,
                                  const size_t test_top_x_configurations);

  // Retrieves the parameters of the best tuning result
  std::unordered_map<std::string, size_t> GetBestResult() const;

  // Prints the results of the tuning either to screen (stdout) or to a specific output-file.
  // Returns the execution time in miliseconds.
  double PUBLIC_API PrintToScreen() const;
  void PUBLIC_API PrintFormatted() const;
  void PUBLIC_API PrintJSON(const std::string &filename,
                            const std::vector<std::pair<std::string,std::string>> &descriptions) const;
  void PUBLIC_API PrintToFile(const std::string &filename) const;

  // Disables all further printing to stdout
  void PUBLIC_API SuppressOutput();

  // Changes the number of times each kernel should be run. Used for averaging execution times.
  void PUBLIC_API SetNumRuns(const size_t num_runs);

 private:

  // This implements the pointer to implementation idiom (pimpl) and hides all private functions and
  // member variables.
  std::unique_ptr<TunerImpl> pimpl;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_CLTUNE_H_
#endif
