
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the simple.cc example
//
// =================================================================================================

// Reference implementation of the matrix-vector multiplication example. Note: this kernel assumes
// that matrix A is pre-transposed.
__kernel void matvec_reference(const int kSizeM, const int kSizeN,
                               const __global float* mat_a,
                               const __global float* vec_x,
                               __global float* vec_y) {

  // Thread identifier
  const int i = get_global_id(0); // From 0 to kSizeM-1

  // Computes a single value
  float result = 0.0f;
  for (int j=0; j<kSizeN; ++j) {
    result += mat_a[j*kSizeM + i] * vec_x[j];
  }

  // Stores the result
  vec_y[i] = result;
}

// =================================================================================================
