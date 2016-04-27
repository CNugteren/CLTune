
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an example OpenCL kernel as part of the simple_kernel.cc example.
//
// =================================================================================================

__kernel void vector_add(const int n, __global float *a, __global float *b, __global float *c) {
  const int i = get_global_id(0);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// =================================================================================================
