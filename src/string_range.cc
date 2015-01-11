
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// The CLTune project follows the Google C++ styleguide and uses a tab-size of two spaces and a
// max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the StringRange class (see the header for information about the class).
//
// =================================================================================================

#include "tuner/internal/string_range.h"

#include <string>

namespace cltune {
// =================================================================================================

StringRange::StringRange() :
  sizes_({"1", "1", "1"}) {
};
StringRange::StringRange(std::string x) :
  sizes_({x, "1", "1"}) {
};
StringRange::StringRange(std::string x, std::string y) :
  sizes_({x, y, "1"}) {
};
StringRange::StringRange(std::string x, std::string y, std::string z) :
  sizes_({x, y, z}) {
};

// =================================================================================================
} // namespace cltune
