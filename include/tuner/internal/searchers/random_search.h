
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements a random-search algorithm, testing the configurations randomly. However,
// it does not consider the same configuration twice. It is derived from the basic search class.
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

#ifndef CLBLAS_TUNER_SEARCHERS_RANDOM_SEARCH_H_
#define CLBLAS_TUNER_SEARCHERS_RANDOM_SEARCH_H_

#include <vector>

#include "tuner/internal/searcher.h"

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class RandomSearch: public Searcher {
 public:

  // Takes additionally a fraction of configurations to try (1.0 == full search)
  RandomSearch(const Configurations &configurations, const float fraction);

  // Retrieves the next configuration to test
  virtual KernelInfo::Configuration GetConfiguration() override;

  // Calculates the next index
  virtual void CalculateNextIndex() override;

  // Retrieves the total number of configurations to try
  virtual size_t NumConfigurations() override;

 private:
    float fraction_;
};

// =================================================================================================
} // namespace cltune

// CLBLAS_TUNER_SEARCHERS_RANDOM_SEARCH_H_
#endif
