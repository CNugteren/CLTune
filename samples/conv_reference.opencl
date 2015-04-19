
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the conv.cc example. This assumes that
// the input matrix is bigger than the output matrix, as it already has padding on the borders. So
// no check is needed within the kernel. This also assumes the workgroup dimension is a multiple
// of the matrix sizes.
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


// Filter settings
#include "../samples/conv.h"

// The filter values (max 7 x 7)
__constant float coeff[FS][FS] = {
  {1/98.0, 1/98.0, 1/98.0, 2/98.0, 1/98.0, 1/98.0, 1/98.0},
  {1/98.0, 1/98.0, 1/98.0, 2/98.0, 1/98.0, 1/98.0, 1/98.0},
  {1/98.0, 1/98.0, 2/98.0, 4/98.0, 2/98.0, 1/98.0, 1/98.0},
  {2/98.0, 2/98.0, 4/98.0, 8/98.0, 4/98.0, 2/98.0, 2/98.0},
  {1/98.0, 1/98.0, 2/98.0, 4/98.0, 2/98.0, 1/98.0, 1/98.0},
  {1/98.0, 1/98.0, 1/98.0, 2/98.0, 1/98.0, 1/98.0, 1/98.0},
  {1/98.0, 1/98.0, 1/98.0, 2/98.0, 1/98.0, 1/98.0, 1/98.0},
};

// =================================================================================================

// Reference implementation of the 2D convolution example
__kernel void conv_reference(const int size_x, const int size_y,
                             const __global float* src,
                             __global float* dest) {

  // Thread identifiers
  const int tid_x = get_global_id(0); // From 0 to size_x-1
  const int tid_y = get_global_id(1); // From 0 to size_y-1

  // Initializes the accumulation register
  float acc = 0.0f;

  // Loops over the neighbourhood
  for (int fx=-HFS; fx<=HFS; ++fx) {
    for (int fy=-HFS; fy<=HFS; ++fy) {
      const int index_x = tid_x + HFS + fx;
      const int index_y = tid_y + HFS + fy;

      // Performs the accumulation
      float coefficient = coeff[fy+HFS][fx+HFS];
      acc += coefficient * src[index_y*size_x + index_x];
    }
  }

  // Computes and stores the result
  dest[tid_y*size_x + tid_x] = acc / (FS * FS);
}

// =================================================================================================
