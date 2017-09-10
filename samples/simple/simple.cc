
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file demonstrates the usage of CLTune with a simple vector-add example. The tuning parameter
// for this example is the work-group/thread-block size.
//
// =================================================================================================

#include <vector>

// Includes the tuner library
#include "cltune.h"

// =================================================================================================

int main() {

  // Sets the filenames of the kernel (either OpenCL or CUDA)
  #ifdef USE_OPENCL
    auto kernel_file = std::vector<std::string>{"../samples/simple/simple_kernel.opencl"};
  #else
    auto kernel_file = std::vector<std::string>{"../samples/simple/simple_kernel.cu"};
  #endif

  // Vector dimension
  const auto kVectorSize = size_t{16*1024*1024};

  // Creates the vectors and fills them with some example data
  std::vector<float> vec_a(kVectorSize, 1.0f);
  std::vector<float> vec_b(kVectorSize, 2.0f);
  std::vector<float> vec_c(kVectorSize, 0.0f);

  // Initializes the tuner (platform 0, device 0)
  cltune::Tuner tuner(size_t{0}, size_t{0});

  // Adds the kernel. The total number of threads (the global size) is equal to 'kVectorSize', and
  // the base number of threads per work-group/thread-block (the local size) is 1. This number is
  // then multiplied by the 'GROUP_SIZE' parameter, which can take any of the specified values.
  const auto id = tuner.AddKernel(kernel_file, "vector_add", {kVectorSize}, {1});
  tuner.AddParameter(id, "GROUP_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048});
  tuner.MulLocalSize(id, {"GROUP_SIZE"});

  // Sets the function's arguments
  tuner.AddArgumentScalar(static_cast<int>(kVectorSize));
  tuner.AddArgumentInput(vec_a);
  tuner.AddArgumentInput(vec_b);
  tuner.AddArgumentOutput(vec_c);

  // Starts the tuner
  tuner.SetNumRuns(10);
  tuner.Tune();

  // Prints the results to screen
  tuner.PrintToScreen();
  tuner.PrintJSON("test.json", {});
  return 0;
}

// =================================================================================================
