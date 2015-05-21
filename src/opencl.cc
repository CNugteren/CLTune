
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file implements the OpenCL class (see the header for information about the class).
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
#include "internal/opencl.h"

namespace cltune {
// =================================================================================================

// Messages printed to stdout (in colours)
const std::string OpenCL::kMessageFull = "\x1b[32m[==========]\x1b[0m";

// =================================================================================================

// Gets a list of all platforms/devices and chooses the selected ones. Initializes OpenCL and also
// downloads properties of the device for later use.
OpenCL::OpenCL(const size_t platform_id, const size_t device_id):
    suppress_output_{false},
    platform_(Platform(platform_id)),
    device_(Device(platform_, kDeviceType, device_id)),
    context_(Context(device_)),
    queue_(CommandQueue(context_, device_)) {

  // Prints the device name
  if (!suppress_output_) {
    fprintf(stdout, "\n%s Initializing OpenCL on platform %lu device %lu\n",
            kMessageFull.c_str(), platform_id, device_id);
    auto opencl_version = device_.Version();
    auto device_name = device_.Name();
    fprintf(stdout, "%s Device name: '%s' (%s)\n", kMessageFull.c_str(),
            device_name.c_str(), opencl_version.c_str());
  }
}

// Releases the OpenCL objects
OpenCL::~OpenCL() {
}

// =================================================================================================
} // namespace cltune
