
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains a base class for search algorithms. It is meant to be inherited by other less
// abstract search algorithms, such as full search or a random search. The pure virtual functions
// declared here are customised in the derived classes. This class stores all configurations which
// could be examined, and receives feedback from the tuner in the form of execution time.
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

#ifndef CLTUNE_SEARCHER_H_
#define CLTUNE_SEARCHER_H_

#include <vector>
#include <chrono>

#include "internal/kernel_info.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class Searcher {
 public:

  // Short-hand for a list of configurations
  using Configurations = std::vector<KernelInfo::Configuration>;

  // Base constructor
  Searcher(const Configurations &configurations);
  virtual ~Searcher() { }

  // Pushes feedback (in the form of execution time) from the tuner to the search algorithm
  virtual void PushExecutionTime(const double execution_time);

  // Prints the log of the search process
  void PrintLog(FILE* fp) const;

  // Pure virtual functions: these are overriden by the derived classes
  virtual KernelInfo::Configuration GetConfiguration() = 0;
  virtual void CalculateNextIndex() = 0;
  virtual size_t NumConfigurations() = 0;

 protected:

  // Pseudo-random seed based on the time
  unsigned int RandomSeed() const {
    // std::random_device rd;
    // return rd();
    return static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
  }

  // Protected member variables accessible by derived classes
  Configurations configurations_;
  std::vector<double> execution_times_;
  std::vector<size_t> explored_indices_;
  size_t index_;
};

// =================================================================================================
} // namespace cltune

// CLTUNE_SEARCHER_H_
#endif
