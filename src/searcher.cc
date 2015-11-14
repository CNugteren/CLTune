
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the Searcher class (see the header for information about the class).
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

// The corresponding header file
#include "internal/searcher.h"

#include <limits>

namespace cltune {
// =================================================================================================

// Simple base-class constructor
Searcher::Searcher(const Configurations &configurations):
    configurations_(configurations),
    execution_times_(configurations.size(), std::numeric_limits<double>::max()),
    explored_indices_(),
    index_(0) {
}

// Adds the resulting execution time to the back of the execution times vector. Also stores the
// index value (to keep track of which indices are explored).
void Searcher::PushExecutionTime(const double execution_time) {
  explored_indices_.push_back(index_);
  execution_times_[index_] = execution_time;
}

// Prints the explored indices and the corresponding execution times to a log(file)
void Searcher::PrintLog(FILE* fp) const {
  fprintf(fp, "step;index;time\n");
  auto step = 0;
  for (auto &explored_index: explored_indices_) {
    fprintf(fp, "%d;%zu;%.3lf\n", step, explored_index, execution_times_[explored_index]);
    ++step;
  }
}

// =================================================================================================
} // namespace cltune
