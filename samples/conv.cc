
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
#include <cmath>
#include <numeric>

// Includes the OpenCL tuner library
#include "cltune.h"

// Helper function to perform an integer division + ceiling (round-up)
int CeilDiv(int a, int b) { return (a + b - 1)/b; }

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(int a, int b) {
  return ((a/b)*b == a) ? true : false;
};

// Constants
constexpr auto kDefaultDevice = 0;
constexpr auto kDefaultSearchMethod = 1;
constexpr auto kDefaultSearchParameter1 = 4;

// Settings (synchronise these with "conv.cc", "conv.opencl" and "conv_reference.opencl")
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

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
  auto coeff = std::vector<float>(FS*FS);

  // Populates data structures
  srand(time(nullptr));
  for (auto &item: mat_a) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_b) { item = 0.0; }

  // Creates the filter coefficients (gaussian blur)
  auto sigma = 1.0f;
  auto mean = FS/2.0f;
  auto sum = 0.0f;
  for (auto x=0; x<FS; ++x) {
    for (auto y=0; y<FS; ++y) {
      auto exponent = -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0));
      coeff[y*FS + x] = exp(exponent) / (2 * M_PI * sigma * sigma);
      sum += coeff[y*FS + x];
    }
  }
  for (auto &item: coeff) { item = item / sum; }

  // ===============================================================================================

  // Initializes the tuner (platform 0, device 'device_id')
  cltune::Tuner tuner(0, device_id);

  // Sets one of the following search methods:
  // 0) Random search
  // 1) Simulated annealing
  // 2) Particle swarm optimisation (PSO)
  // 3) Full search
  auto fraction = 1/32.0f;
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
  tuner.AddParameter(id, "PADDING", {0, 1});

  // Introduces a helper parameter to compute the proper number of threads for the LOCAL == 2 case.
  // In this case, the workgroup size (TBX by TBY) is extra large (TBX_XL by TBY_XL) because it uses
  // extra threads to compute the halo threads. How many extra threads are needed is dependend on
  // the filter size. Here we support a the TBX and TBY size plus up to 10 extra threads.
  auto integers = {8,9,10,11,12,13,14,15,
                   16,17,18,19,20,21,22,23,24,25,26,
                   32,33,34,35,36,37,38,39,40,41,42,
                   64,65,66,67,68,69,70,71,72,73,74};
  tuner.AddParameter(id, "TBX_XL", integers);
  tuner.AddParameter(id, "TBY_XL", integers);
  auto HaloThreads = [] (std::vector<int> v) {
    if (v[0] == 2) { return (v[1] == v[2] + CeilDiv(2*HFS,v[3])); } // With halo threads
    else           { return (v[1] == v[2]); }                       // Without halo threads
  };
  tuner.AddConstraint(id, HaloThreads, {"LOCAL", "TBX_XL", "TBX", "WPTX"});
  tuner.AddConstraint(id, HaloThreads, {"LOCAL", "TBY_XL", "TBY", "WPTY"});

  // Sets the constrains on the vector size
  auto VectorConstraint = [] (std::vector<int> v) {
    if (v[0] == 2) { return IsMultiple(v[2],v[1]) && IsMultiple(2*HFS,v[1]); }
    else           { return IsMultiple(v[2],v[1]); }
  };
  tuner.AddConstraint(id, VectorConstraint, {"LOCAL", "VECTOR", "WPTX"});

  // Sets padding to zero in case local memory is not used
  auto PaddingConstraint = [] (std::vector<int> v) { return (v[1] == 0 || v[0] != 0); };
  tuner.AddConstraint(id, PaddingConstraint, {"LOCAL", "PADDING"});

  // Sets the constraints for local memory size limitations
  auto LocalMemorySize = [] (std::vector<int> v) {
    if (v[0] != 0) { return ((v[3]*v[4] + 2*HFS) * (v[1]*v[2] + 2*HFS + v[5]))*sizeof(float); }
    else           { return 0UL; }
  };
  tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"LOCAL", "TBX", "WPTX", "TBY", "WPTY", "PADDING"});

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
  tuner.AddArgumentInput(coeff);
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
