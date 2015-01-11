
// =================================================================================================
// This file is part of the CLTune project. The project is licensed under the MIT license by
// SURFsara, (c) 2014.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the simple.cc example
//
// =================================================================================================

// Parameters set by the tuner
// UNROLL: Unroll factor of loop over kSizeN

// Manually unrolled version of the matrix-vector multiplication example. Note: this kernel assumes
// that matrix A is pre-transposed.
__kernel void matvec_unroll(const int kSizeM, const int kSizeN,
                            const __global float* mat_a,
                            const __global float* vec_x,
                            __global float* vec_y) {

  // Thread identifier
  const int i = get_global_id(0); // From 0 to kSizeM-1

  // Initializes the accumulation register
  float result = 0.0f;

  // Regular code
  #if UNROLL == 1
    for (int j=0; j<kSizeN; ++j) {
      result += mat_a[j*kSizeM + i] * vec_x[j];
    }

  // Unrolled by a factor 2
  #elif UNROLL == 2
    for (int j=0; j<kSizeN; j+=UNROLL) {
      result += mat_a[(j + 0)*kSizeM + i] * vec_x[j + 0];
      result += mat_a[(j + 1)*kSizeM + i] * vec_x[j + 1];
    }

  // Unrolled by a factor 4
  #elif UNROLL == 4
    for (int j=0; j<kSizeN; j+=UNROLL) {
      result += mat_a[(j + 0)*kSizeM + i] * vec_x[j + 0];
      result += mat_a[(j + 1)*kSizeM + i] * vec_x[j + 1];
      result += mat_a[(j + 2)*kSizeM + i] * vec_x[j + 2];
      result += mat_a[(j + 3)*kSizeM + i] * vec_x[j + 3];
    }
  #endif

  // Stores the result
  vec_y[i] = result;
}

// =================================================================================================
