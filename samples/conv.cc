
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file demonstrates the usage of CLTune with 2D convolution and advanced search techniques
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

#include <iostream>
#include <sstream>
#include <vector>

// Includes the OpenCL tuner library
#include "cltune.h"

// Constants
constexpr auto kDefaultDevice = 0;
constexpr auto kDefaultSearchMethod = 1;
constexpr auto kDefaultSearchParameter1 = 4;

// Settings (device)
constexpr auto kMaxLocalThreads = 1024;
constexpr auto kMaxLocalMemory = 32*1024;

// Settings (also change these in conv.cc, conv.opencl, and conv_reference.opencl!!)
#define HFS (3)        // Half filter size (synchronise with other files)
#define FS (HFS+HFS+1) // Filter size
#define FA (FS*FS)     // Filter area

// Settings (sizes)
constexpr auto kSizeX = 8192; // Matrix dimension X
constexpr auto kSizeY = 4096; // Matrix dimension Y

// =================================================================================================

// Example showing how to tune an OpenCL 2D convolution kernel
int main(int argc, char* argv[]) {

  // Selects the device, the search method and its first parameter. These parameters are all
  // optional and are thus also given default values.
  auto device_id = kDefaultDevice;
  auto method = kDefaultSearchMethod;
  auto search_param_1 = kDefaultSearchParameter1;
  if (argc >= 2) {
    device_id = std::stoi(std::string{argv[1]});
    if (argc >= 3) {
      method = std::stoi(std::string{argv[2]});
      if (argc >= 4) {
        search_param_1 = std::stoi(std::string{argv[3]});
      }
    }
  }

  // Creates data structures
  auto mat_a = std::vector<float>((2*HFS+kSizeX)*(2*HFS+kSizeY));
  auto mat_b = std::vector<float>(kSizeX*kSizeY);

  // Populates data structures
  srand(time(nullptr));
  for (auto &item: mat_a) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_b) { item = 0.0; }

  // Initializes the tuner (platform 0, device 'device_id')
  cltune::Tuner tuner(0, device_id);

  // Sets one of the following search methods:
  // 0) Random search
  // 1) Simulated annealing
  // 2) Particle swarm optimisation (PSO)
  // 3) Full search
  auto fraction = 1/16.0f;
  if      (method == 0) { tuner.UseRandomSearch(fraction); }
  else if (method == 1) { tuner.UseAnnealing(fraction, search_param_1); }
  else if (method == 2) { tuner.UsePSO(fraction, search_param_1, 0.4, 0.0, 0.4); }
  else                  { tuner.UseFullSearch(); }

  // Outputs the search process to a file
  tuner.OutputSearchLog("search_log.txt");

  // ===============================================================================================

  // Adds a heavily tuneable kernel and some example parameter values
  auto id = tuner.AddKernel("../samples/conv.opencl", "conv", {kSizeX, kSizeY}, {1, 1});
  tuner.AddParameter(id, "TBX", {8, 16, 32, 64});
  tuner.AddParameter(id, "TBY", {8, 16, 32, 64});
  tuner.AddParameter(id, "LOCAL", {0, 1, 2});
  tuner.AddParameter(id, "WPTX", {1, 2, 4, 8});
  tuner.AddParameter(id, "WPTY", {1, 2, 4, 8});
  tuner.AddParameter(id, "VECTOR", {1, 2, 4});
  tuner.AddParameter(id, "UNROLL_FACTOR", {1, FS});

  // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case
  tuner.AddParameter(id, "TBX_XL", {8, 8+2*HFS, 16, 16+2*HFS, 32, 32+2*HFS, 64, 64+2*HFS});
  tuner.AddParameter(id, "TBY_XL", {8, 8+2*HFS, 16, 16+2*HFS, 32, 32+2*HFS, 64, 64+2*HFS});
  auto HaloThreads = [] (std::vector<int> v) {
    if (v[0] == 2) { return (v[1] == v[2] + 2*HFS); } // With halo threads
    else           { return (v[1] == v[2]); }          // Without halo threads
  };
  tuner.AddConstraint(id, HaloThreads, {"LOCAL", "TBX_XL", "TBX"});
  tuner.AddConstraint(id, HaloThreads, {"LOCAL", "TBY_XL", "TBY"});

  // Sets the constrains on the vector size
  auto VectorConstraint = [] (std::vector<int> v) { return (v[0] <= v[1]); };
  tuner.AddConstraint(id, VectorConstraint, {"VECTOR", "WPTX"});

  // Set the constraints for architecture limitations
  auto LocalWorkSize = [] (std::vector<int> v) { return (v[0]*v[1] <= kMaxLocalThreads); };
  auto LocalMemorySize = [] (std::vector<int> v) {
    return (v[0]*v[1]*v[2]*v[3]*sizeof(float) <= kMaxLocalMemory);
  };
  tuner.AddConstraint(id, LocalWorkSize, {"TBX_XL", "TBY_XL"});
  tuner.AddConstraint(id, LocalMemorySize, {"TBX_XL", "WPTX", "TBY_XL", "WPTY"});

  // Modifies the thread-sizes based on the parameters
  tuner.MulLocalSize(id, {"TBX_XL", "TBY_XL"});
  tuner.MulGlobalSize(id, {"TBX_XL", "TBY_XL"});
  tuner.DivGlobalSize(id, {"TBX", "TBY"});
  tuner.DivGlobalSize(id, {"WPTX", "WPTY"});

  // ===============================================================================================

  // Sets the tuner's golden reference function. This kernel contains the reference code to which
  // the output is compared. Supplying such a function is not required, but it is necessary for
  // correctness checks to be enabled.
  tuner.SetReference("../samples/conv_reference.opencl", "conv_reference", {kSizeX, kSizeY}, {8,8});

  // Sets the function's arguments. Note that all kernels have to accept (but not necessarily use)
  // all input arguments.
  tuner.AddArgumentScalar(kSizeX);
  tuner.AddArgumentScalar(kSizeY);
  tuner.AddArgumentInput(mat_a);
  tuner.AddArgumentOutput(mat_b);

  // Starts the tuner
  tuner.Tune();

  // Prints the results to screen and to file
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintToFile("output.csv");

  // Also prints the performance of the best-case in terms of GB/s and GFLOPS
  constexpr auto kMB = (sizeof(float)*2*(long)kSizeX*(long)kSizeY) / (1.0e6);
  constexpr auto kMFLOPS = ((1+2*FS*FS)*(long)kSizeX*(long)kSizeY) / (1.0e6);
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.1lf GB/s or %1.lf GFLOPS\n",
           time_ms, kMB/time_ms, kMFLOPS/time_ms);
  }

  // End of the tuner example
  return 0;
}

// =================================================================================================
