
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// The CLTune project follows the Google C++ styleguide and uses a tab-size of two spaces and a
// max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file demonstrates the usage of CLTune with a more advanced matrix-multiplication example.
// This matrix-matrix multiplication is also heavily tuned and competes performance-wise with the
// clBLAS library.
//
// =================================================================================================

#include <iostream>
#include <sstream>
#include <vector>

// Includes the OpenCL tuner library
#include "tuner/tuner.h"

// =================================================================================================

// Example showing how to tune an OpenCL SGEMM matrix-multiplication kernel. This assumes that
// matrix B is pre-transposed, alpha equals 1 and beta equals 0: C = A * B^T
int main(int argc, char* argv[]) {

  // Creates data structures
  const int kSizeM = 256;
  const int kSizeN = 512;
  const int kSizeK = 1024;

  // Creates input matrices
  std::vector<float> mat_a(kSizeM*kSizeK);
  std::vector<float> mat_b(kSizeN*kSizeK);
  std::vector<float> mat_c(kSizeM*kSizeN);

  // Populates input data structures
  srand(time(NULL));
  for (auto &item: mat_a) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_b) { item = (float)rand() / (float)RAND_MAX; }
  for (auto &item: mat_c) { item = 0.0; }

  // Initializes the tuner (platform 0, device 1)
  cltune::Tuner tuner(0, 0);
  
  // Adds a heavily tuneable kernel and some example parameter values. Others can be added, but for
  // this example this already leads to plenty of kernels to test.
  size_t id = tuner.AddKernel("../samples/gemm_fast.opencl", "gemm_fast", {kSizeM, kSizeN}, {1, 1});
  tuner.AddParameter(id, "MWG", {64, 128});
  tuner.AddParameter(id, "NWG", {64, 128});
  tuner.AddParameter(id, "KWG", {16});
  tuner.AddParameter(id, "MDIMC", {16});
  tuner.AddParameter(id, "NDIMC", {16});
  tuner.AddParameter(id, "MDIMA", {32});
  tuner.AddParameter(id, "NDIMB", {32});
  tuner.AddParameter(id, "KWI", {8});
  tuner.AddParameter(id, "VWM", {1, 2});
  tuner.AddParameter(id, "VWN", {1, 2});
  tuner.AddParameter(id, "STRM", {1});
  tuner.AddParameter(id, "STRN", {1});
  tuner.AddParameter(id, "SA", {0, 1});
  tuner.AddParameter(id, "SB", {0, 1});

  // Tests single precision (SGEMM)
  tuner.AddParameter(id, "PRECISION", {32});

  // Sets constraints: Requirement for unrolling the KWG loop
  tuner.AddConstraint(id, "KWG", cltune::kMultipleOf, "KWI");

  // Sets constraints: Required for integer MWI and NWI
  tuner.AddConstraint(id, "MWG", cltune::kMultipleOf, "MDIMC", cltune::kMultipliedBy, "VWM");
  tuner.AddConstraint(id, "NWG", cltune::kMultipleOf, "NDIMC", cltune::kMultipliedBy, "VWN");

  // Sets constraints: Required for integer MWIA and NWIB
  tuner.AddConstraint(id, "MWG", cltune::kMultipleOf, "MDIMA", cltune::kMultipliedBy, "VWM");
  tuner.AddConstraint(id, "NWG", cltune::kMultipleOf, "NDIMB", cltune::kMultipliedBy, "VWN");

  // Sets constraints: KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
  tuner.AddConstraint(id, "KWG", cltune::kMultipleOf, "MDIMC", cltune::kMultipliedBy, "NDIMC",
                      cltune::kDividedBy, "MDIMA");
  tuner.AddConstraint(id, "KWG", cltune::kMultipleOf, "MDIMC", cltune::kMultipliedBy, "NDIMC",
                      cltune::kDividedBy, "NDIMB");

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
  tuner.AddArgumentScalar<int>(kSizeM);
  tuner.AddArgumentScalar<int>(kSizeN);
  tuner.AddArgumentScalar<int>(kSizeK);
  tuner.AddArgumentInput<float>(mat_a);
  tuner.AddArgumentInput<float>(mat_b);
  tuner.AddArgumentOutput<float>(mat_c);

  // Starts the tuner
  tuner.Tune();

  // Prints the results to screen and to file
  double time_ms = tuner.PrintToScreen();
  tuner.PrintToFile("output.csv");

  // Also prints the performance of the best-case in terms of GFLOPS
  const double kGFLOP = ((long)kSizeM * (long)kSizeN * (long)kSizeK * 2) / (1000.0*1000.0*1000.0);
  if (time_ms != 0.0) {
    printf("[ -------> ] %.1lf ms or %.3lf GFLOPS\n", time_ms, 1000*kGFLOP/time_ms);
  }

  // End of the tuner example
  return 0;
}

// =================================================================================================
