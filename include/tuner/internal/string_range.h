
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains the StringRange class, which is the same as the OpenCL cl::NDRange class, but
// with string-based representations of the dimensions instead.
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

#ifndef CLBLAS_TUNER_STRING_RANGE_H_
#define CLBLAS_TUNER_STRING_RANGE_H_

#include <string>
#include <vector>

namespace cltune {
// =================================================================================================

// See comment at top of file for a description of the class
class StringRange {
 public:
  // Initializes the class with 0, 1, 2, or 3 dimensions. These constructors are not explicit
  // because they are used by clients in the form of initializer lists when for example calling
  // cltuner::MulGlobalSize.
  StringRange();
  StringRange(std::string x);
  StringRange(std::string x, std::string y);
  StringRange(std::string x, std::string y, std::string z);

  // Accessor of sizes per dimension (getter)
  std::string sizes(int id) const { return sizes_[id]; }

 private:

  // Member variables
  std::vector<std::string> sizes_;
};

// =================================================================================================
} // namespace cltune

// CLBLAS_TUNER_STRING_RANGE_H_
#endif
