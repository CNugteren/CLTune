
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the gemm.cc example.
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

// Reference implementation of the matrix-matrix multiplication example. Note: this kernel assumes
// that matrix B is pre-transposed.
__kernel void gemm_reference(const int kSizeM, const int kSizeN, const int kSizeK,
                             const __global float* mat_a,
                             const __global float* mat_b,
                             __global float* mat_c) {

  // Thread identifiers
  const int row = get_global_id(0); // From 0 to kSizeM-1
  const int col = get_global_id(1); // From 0 to kSizeN-1

  // Computes a single value
  float result = 0.0f;
  for (int k=0; k<kSizeK; k++) {
    float mat_a_val = mat_a[k*kSizeM + row];
    float mat_b_val = mat_b[k*kSizeN + col];
    result += mat_a_val * mat_b_val;
  }

  // Stores the result
  mat_c[col*kSizeM + row] = result;
}

// =================================================================================================
