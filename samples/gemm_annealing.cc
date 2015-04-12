
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file demonstrates the usage of CLTune with a more advanced matrix-multiplication example.
// This matrix-matrix multiplication is also heavily tuned and competes performance-wise with the
// clBLAS library.
// In contrast to the regular 'gemm' example, the 'gemm_annealing' example searchers through a much
// larger parameter space, but uses simulated annealing instead of full search.
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

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(int a, int b) {
  return ((a/b)*b == a) ? true : false;
};

// =================================================================================================

// Example showing how to tune an OpenCL SGEMM matrix-multiplication kernel. This assumes that
// matrix B is pre-transposed, alpha equals 1 and beta equals 0: C = A * B^T
int main() {

  // Creates data structures
  constexpr auto kSizeM = 2048;
  constexpr auto kSizeN = 2048;
  constexpr auto kSizeK = 2048;

  // Creates input matrices
  auto mat_a = std::vector<float>(kSizeM*kSizeK);
  auto mat_b = std::vector<float>(kSizeN*kSizeK);
  auto mat_c = std::vector<float>(kSizeM*kSizeN);

  // Populates input data structures
  srand(time(nullptr));
  for (auto &item: mat_a) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_b) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_c) { item = 0.0; }

  // Initializes the tuner (platform 0, device 0)
  cltune::Tuner tuner(0, 1);

  // Configures the tuner to select the simulated annealing search method, setting the fraction of
  // the search space to explore (1/64th) and the maximum annealing temperature (relative to the
  // execution time in miliseconds).
  tuner.UseAnnealing(1/64.0, 4.0);

  // Outputs the search process to a file
  tuner.OutputSearchLog("search_log.txt");
  
  // Adds a heavily tuneable kernel and some example parameter values. Others can be added, but for
  // this example this already leads to plenty of kernels to test.
  auto id = tuner.AddKernel("../samples/gemm_fast.opencl", "gemm_fast", {kSizeM, kSizeN}, {1, 1});
  tuner.AddParameter(id, "MWG", {64, 128});
  tuner.AddParameter(id, "NWG", {64, 128});
  tuner.AddParameter(id, "KWG", {16, 32});
  tuner.AddParameter(id, "MDIMC", {8, 16, 32});
  tuner.AddParameter(id, "NDIMC", {8, 16, 32});
  tuner.AddParameter(id, "MDIMA", {16, 32});
  tuner.AddParameter(id, "NDIMB", {16, 32});
  tuner.AddParameter(id, "KWI", {8});
  tuner.AddParameter(id, "VWM", {1, 2, 4});
  tuner.AddParameter(id, "VWN", {1, 2, 4});
  tuner.AddParameter(id, "STRM", {1});
  tuner.AddParameter(id, "STRN", {1});
  tuner.AddParameter(id, "SA", {0, 1});
  tuner.AddParameter(id, "SB", {0, 1});

  // Tests single precision (SGEMM)
  tuner.AddParameter(id, "PRECISION", {32});

  // Sets constraints: Set-up the constraints functions to use. The constraints require a function
  // object (in this case a lambda) which takes a vector of tuning parameter values and returns
  // a boolean value whether or not the tuning configuration is legal. In this case, the helper
  // function 'IsMultiple' is employed for convenience. In the calls to 'AddConstraint' below, the
  // vector of parameter names (as strings) matches the input integer vector of the lambda's.
  auto MultipleOfX = [] (std::vector<int> v) { return IsMultiple(v[0], v[1]); };
  auto MultipleOfXMulY = [] (std::vector<int> v) { return IsMultiple(v[0], v[1]*v[2]); };
  auto MultipleOfXMulYDivZ = [] (std::vector<int> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };

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

  // Modifies the thread-sizes (both global and local) based on the parameters
  tuner.MulLocalSize(id, {"MDIMC", "NDIMC"});
  tuner.MulGlobalSize(id, {"MDIMC", "NDIMC"});
  tuner.DivGlobalSize(id, {"MWG", "NWG"});

  // Sets the tuner's golden reference function. This kernel contains the reference code to which
  // the output is compared. Supplying such a function is not required, but it is necessarily for
  // correctness checks to be enabled.
  tuner.SetReference("../samples/gemm_reference.opencl", "gemm_reference", {kSizeM, kSizeN}, {8,8});

  // Sets the function's arguments. Note that all kernels have to accept (but not necessarily use)
  // all input arguments.
  tuner.AddArgumentScalar(kSizeM);
  tuner.AddArgumentScalar(kSizeN);
  tuner.AddArgumentScalar(kSizeK);
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
  constexpr auto kGFLOP = (2*(long)kSizeM*(long)kSizeN*(long)kSizeK) / (1000.0*1000.0*1000.0);
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.3lf GFLOPS\n", time_ms, 1000*kGFLOP/time_ms);
  }

  // End of the tuner example
  return 0;
}

// =================================================================================================
