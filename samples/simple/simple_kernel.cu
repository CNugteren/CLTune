
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an example CUDA kernel as part of the simple_kernel.cc example.
//
// =================================================================================================

extern "C" __global__ void vector_add(const int n, float *a, float *b, float *c) {
  const int i = blockIdx.x * GROUP_SIZE + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// =================================================================================================
