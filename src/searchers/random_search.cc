
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the RandomSearch class (see the header for information about the class).
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

#include "tuner/internal/searchers/random_search.h"

#include <algorithm>

namespace cltune {
// =================================================================================================

// Randomizes the configurations list
RandomSearch::RandomSearch(const Configurations &configurations, const float fraction):
    Searcher(configurations),
    fraction_(fraction) {
  std::random_shuffle(configurations_.begin(), configurations_.end());
}

// =================================================================================================

// Returns the next configuration (vector of configurations is already shuffled randomly)
KernelInfo::Configuration RandomSearch::GetConfiguration() {
  return configurations_[index_];
}

// Calculates the index of the next configuration to test
void RandomSearch::CalculateNextIndex() {
  ++index_;
}

// The number of configurations is equal to all possible configurations
size_t RandomSearch::NumConfigurations() {
  return configurations_.size()*fraction_;
}

// =================================================================================================
} // namespace cltune
