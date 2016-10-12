
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the usage of CLTune with a convolution example. The tuning parameters
// include the work-group/thread-block size and the vectorisation and thread-coarsening factors.
//
// =================================================================================================

// Settings (synchronise these among all "conv_simple.*" files)
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

#include <vector>

// Includes the tuner library
#include "cltune.h"

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b) {
  return ((a/b)*b == a) ? true : false;
};

// =================================================================================================

int main() {

  // Sets the filenames of the kernel (either OpenCL or CUDA)
  #ifdef USE_OPENCL
    auto kernel_file = std::vector<std::string>{"../samples/conv_simple/conv_simple_kernel.opencl"};
  #else
    auto kernel_file = std::vector<std::string>{"../samples/conv_simple/conv_simple_kernel.cu"};
  #endif

  // Input/output sizes
  const auto kSizeX = size_t{8192}; // Matrix dimension X
  const auto kSizeY = size_t{4096}; // Matrix dimension Y

  // Creates the input/output matrices and fills them with some example data
  std::vector<float> mat_a(kSizeX*kSizeY, 2.0f);
  std::vector<float> mat_b(kSizeX*kSizeY, 0.0f);

  // Creates the filter coefficients and fills them with example constant values
  std::vector<float> coeff(FS*FS, 0.05f);

  // Initializes the tuner (platform 0, device 0)
  cltune::Tuner tuner(size_t{0}, size_t{0});

  // ===============================================================================================

  // Adds a tuneable kernel and some example parameter values
  auto id = tuner.AddKernel(kernel_file, "conv", {kSizeX, kSizeY}, {1, 1});
  tuner.AddParameter(id, "TBX", {8, 16, 32});
  tuner.AddParameter(id, "TBY", {8, 16, 32});
  tuner.AddParameter(id, "WPTX", {1, 2, 4});
  tuner.AddParameter(id, "WPTY", {1, 2, 4});
  tuner.AddParameter(id, "VECTOR", {1, 2, 4});

  // Sets the constrains on the vector size
  auto VectorConstraint = [] (std::vector<size_t> v) { return IsMultiple(v[1],v[0]); };
  tuner.AddConstraint(id, VectorConstraint, {"VECTOR", "WPTX"});

  // Modifies the thread-sizes based on the parameters
  tuner.MulLocalSize(id, {"TBX", "TBY"});
  tuner.MulGlobalSize(id, {"TBX", "TBY"});
  tuner.DivGlobalSize(id, {"TBX", "TBY"});
  tuner.DivGlobalSize(id, {"WPTX", "WPTY"});

  // ===============================================================================================

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(kSizeX));
  tuner.AddArgumentScalar(static_cast<int>(kSizeY));
  tuner.AddArgumentInput(mat_a);
  tuner.AddArgumentInput(coeff);
  tuner.AddArgumentOutput(mat_b);

  // Starts the tuner
  tuner.Tune();

  // Prints the results to screen and to file
  tuner.PrintToScreen();
  tuner.PrintToFile("output.csv");

  // End of the tuner example
  return 0;
}

// =================================================================================================
