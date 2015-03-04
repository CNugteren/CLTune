
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the FullSearch class (see the header for information about the class).
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

#include "tuner/internal/searchers/full_search.h"

#include <string>

namespace cltune {
// =================================================================================================

// Calls the base-class constructor directly
FullSearch::FullSearch(const Configurations &configurations):
    Searcher(configurations) {
}

// =================================================================================================

// Returns the next configuration in order
KernelInfo::Configuration& FullSearch::NextConfiguration() {
  ++i;
  return configurations_[i-1];
}

// The number of configurations is equal to all possible configurations
size_t FullSearch::NumConfigurations() {
  return configurations_.size();
}

// =================================================================================================
} // namespace cltune
