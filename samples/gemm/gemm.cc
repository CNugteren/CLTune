
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file demonstrates the usage of CLTune with a more advanced matrix-multiplication example.
// This matrix-matrix multiplication is also heavily tuned and competes performance-wise with the
// clBLAS library.
// In contrast to the regular 'gemm' example, the 'gemm_search_methods' example searchers through a
// much larger parameter space, but uses smart search techniques instead of full search. Examples
// are simulated annealing (the default) and particle swarm optimisation (see below).
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
#include <chrono>
#include <random>

// Includes the OpenCL tuner library
#include "cltune.h"

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) {
  return ((a/b)*b == a) ? true : false;
};

// Constants
constexpr auto kDefaultDevice = size_t{0};
constexpr auto kDefaultSearchMethod = size_t{1};
constexpr auto kDefaultSearchParameter1 = size_t{4};

// Settings (sizes)
constexpr auto kSizeM = size_t{2048};
constexpr auto kSizeN = size_t{2048};
constexpr auto kSizeK = size_t{2048};

// =================================================================================================

// Example showing how to tune an OpenCL SGEMM matrix-multiplication kernel. This assumes that
// matrix B is pre-transposed, alpha equals 1 and beta equals 0: C = A * B^T
int main(int argc, char* argv[]) {

  // Sets the filenames of the OpenCL kernels (optionally automatically translated to CUDA)
  auto gemm_fast = std::vector<std::string>{"../samples/gemm/gemm.opencl"};
  auto gemm_reference = std::vector<std::string>{"../samples/gemm/gemm_reference.opencl"};
  #ifndef USE_OPENCL
    gemm_fast.insert(gemm_fast.begin(), "../samples/cl_to_cuda.h");
    gemm_reference.insert(gemm_reference.begin(), "../samples/cl_to_cuda.h");
  #endif

  // Selects the device, the search method and its first parameter. These parameters are all
  // optional and are thus also given default values.
  auto device_id = kDefaultDevice;
  auto method = kDefaultSearchMethod;
  auto search_param_1 = kDefaultSearchParameter1;
  if (argc >= 2) {
    device_id = static_cast<size_t>(std::stoi(std::string{argv[1]}));
    if (argc >= 3) {
      method = static_cast<size_t>(std::stoi(std::string{argv[2]}));
      if (argc >= 4) {
        search_param_1 = static_cast<size_t>(std::stoi(std::string{argv[3]}));
      }
    }
  }

  // Creates input matrices
  auto mat_a = std::vector<float>(kSizeM*kSizeK);
  auto mat_b = std::vector<float>(kSizeN*kSizeK);
  auto mat_c = std::vector<float>(kSizeM*kSizeN);

  // Create a random number generator
  const auto random_seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(static_cast<unsigned int>(random_seed));
  std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);

  // Populates input data structures
  for (auto &item: mat_a) { item = distribution(generator); }
  for (auto &item: mat_b) { item = distribution(generator); }
  for (auto &item: mat_c) { item = 0.0f; }

  // Initializes the tuner (platform 0, device 'device_id')
  cltune::Tuner tuner(size_t{0}, static_cast<size_t>(device_id));

  // Sets one of the following search methods:
  // 0) Random search
  // 1) Simulated annealing
  // 2) Particle swarm optimisation (PSO)
  // 3) Full search
  auto fraction = 1.0f/2048.0f;
  if      (method == 0) { tuner.UseRandomSearch(fraction); }
  else if (method == 1) { tuner.UseAnnealing(fraction, static_cast<size_t>(search_param_1)); }
  else if (method == 2) { tuner.UsePSO(fraction, static_cast<size_t>(search_param_1), 0.4, 0.0, 0.4); }
  else                  { tuner.UseFullSearch(); }

  // Outputs the search process to a file
  tuner.OutputSearchLog("search_log.txt");
  
  // ===============================================================================================

  // Adds a heavily tuneable kernel and some example parameter values. Others can be added, but for
  // this example this already leads to plenty of kernels to test.
  auto id = tuner.AddKernel(gemm_fast, "gemm_fast", {kSizeM, kSizeN}, {1, 1});
  tuner.AddParameter(id, "MWG", {16, 32, 64, 128});
  tuner.AddParameter(id, "NWG", {16, 32, 64, 128});
  tuner.AddParameter(id, "KWG", {16, 32});
  tuner.AddParameter(id, "MDIMC", {8, 16, 32});
  tuner.AddParameter(id, "NDIMC", {8, 16, 32});
  tuner.AddParameter(id, "MDIMA", {8, 16, 32});
  tuner.AddParameter(id, "NDIMB", {8, 16, 32});
  tuner.AddParameter(id, "KWI", {2, 8});
  tuner.AddParameter(id, "VWM", {1, 2, 4, 8});
  tuner.AddParameter(id, "VWN", {1, 2, 4, 8});
  tuner.AddParameter(id, "STRM", {0, 1});
  tuner.AddParameter(id, "STRN", {0, 1});
  tuner.AddParameter(id, "SA", {0, 1});
  tuner.AddParameter(id, "SB", {0, 1});

  // Tests single precision (SGEMM)
  tuner.AddParameter(id, "PRECISION", {32});

  // Sets constraints: Set-up the constraints functions to use. The constraints require a function
  // object (in this case a lambda) which takes a vector of tuning parameter values and returns
  // a boolean value whether or not the tuning configuration is legal. In this case, the helper
  // function 'IsMultiple' is employed for convenience. In the calls to 'AddConstraint' below, the
  // vector of parameter names (as strings) matches the input integer vector of the lambda's.
  auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
  auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };

  // Sets constraints: Requirement for unrolling the KWG loop
  tuner.AddConstraint(id, MultipleOfX, {"KWG", "KWI"});

  // Sets constraints: Required for integer MWI and NWI
  tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMC", "VWM"});
  tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMC", "VWN"});

  // Sets constraints: Required for integer MWIA and NWIB
  tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMA", "VWM"});
  tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMB", "VWN"});

  // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
  tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "MDIMA"});
  tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "NDIMB"});

  // Sets the constraints for local memory size limitations
  auto LocalMemorySize = [] (std::vector<size_t> v) {
    return (((v[0]*v[1]*v[2]/v[3]) + (v[4]*v[5]*v[6]/v[7]))*sizeof(float));
  };
  tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG", "VWM", "SB", "KWG", "NWG", "VWN"});

  // Modifies the thread-sizes (both global and local) based on the parameters
  tuner.MulLocalSize(id, {"MDIMC", "NDIMC"});
  tuner.MulGlobalSize(id, {"MDIMC", "NDIMC"});
  tuner.DivGlobalSize(id, {"MWG", "NWG"});

  // ===============================================================================================

  // Sets the tuner's golden reference function. This kernel contains the reference code to which
  // the output is compared. Supplying such a function is not required, but it is necessarily for
  // correctness checks to be enabled.
  tuner.SetReference(gemm_reference, "gemm_reference", {kSizeM, kSizeN}, {8,8});

  // Sets the function's arguments. Note that all kernels have to accept (but not necessarily use)
  // all input arguments.
  tuner.AddArgumentScalar(static_cast<int>(kSizeM));
  tuner.AddArgumentScalar(static_cast<int>(kSizeN));
  tuner.AddArgumentScalar(static_cast<int>(kSizeK));
  tuner.AddArgumentInput(mat_a);
  tuner.AddArgumentInput(mat_b);
  tuner.AddArgumentOutput(mat_c);

  // Starts the tuner
  tuner.Tune();

  // Prints the results to screen and to file
  auto time_ms = tuner.PrintToScreen();
  tuner.PrintToFile("output.csv");
  tuner.PrintFormatted();

  // Also prints the performance of the best-case in terms of GFLOPS
  constexpr auto kMGFLOP = (2*kSizeM*kSizeN*kSizeK) * 1.0e-6;
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.3lf GFLOPS\n", time_ms, kMGFLOP/time_ms);
  }

  // End of the tuner example
  return 0;
}

// =================================================================================================
