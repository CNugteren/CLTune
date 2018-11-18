
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

// Settings (synchronise these with "conv.cc", "conv.opencl" and "conv_reference.opencl")
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

// Vector data-types
#if VECTOR == 1
    typedef float floatvec;
#elif VECTOR == 2
    typedef float2 floatvec;
#elif VECTOR == 4
    typedef float4 floatvec;
#elif VECTOR == 8
    typedef float8 floatvec;
#endif

// =================================================================================================

// Initialize the accumulation registers
inline void InitAccRegisters(float acc[WPTY][WPTX]) {
  #pragma unroll
  for (int wx=0; wx<WPTX; ++wx) {
    #pragma unroll
    for (int wy=0; wy<WPTY; ++wy) {
      acc[wy][wx] = 0.0f;
    }
  }
}

// =================================================================================================

// Loads data into local memory
#if LOCAL == 2
inline void LoadLocalFull(__local float *lmem, const int loff,
                          const __global floatvec* src, const int goff,
                          const int gid_x, const int gid_y, const int lid_x, const int lid_y) {

  // Loop over the amount of work per thread
  #pragma unroll
  for (int wx=0; wx<WPTX/VECTOR; ++wx) {
    const int lx = lid_x*WPTX/VECTOR + wx;
    #if WPTX > 0
      if (lx < TBX*WPTX/VECTOR + (2*HFS)/VECTOR)
    #endif
    {
      const int gx = gid_x*WPTX/VECTOR + wx;
      #pragma unroll
      for (int wy=0; wy<WPTY; ++wy) {
        const int ly = lid_y*WPTY + wy;
        #if WPTY > 0
          if (ly < TBY*WPTY + 2*HFS)
        #endif
        {
          const int gy = gid_y*WPTY + wy;

          // Load the data into local memory (WPTX elements per thread)
          floatvec temp = src[gy*goff/VECTOR + gx];
          #if VECTOR == 1
            lmem[(ly)*loff + (lx*VECTOR  )] = temp;
          #elif VECTOR == 2
            lmem[(ly)*loff + (lx*VECTOR  )] = temp.x;
            lmem[(ly)*loff + (lx*VECTOR+1)] = temp.y;
          #elif VECTOR == 4
            lmem[(ly)*loff + (lx*VECTOR  )] = temp.x;
            lmem[(ly)*loff + (lx*VECTOR+1)] = temp.y;
            lmem[(ly)*loff + (lx*VECTOR+2)] = temp.z;
            lmem[(ly)*loff + (lx*VECTOR+3)] = temp.w;
          #elif VECTOR == 8
            lmem[(ly)*loff + (lx*VECTOR  )] = temp.s0;
            lmem[(ly)*loff + (lx*VECTOR+1)] = temp.s1;
            lmem[(ly)*loff + (lx*VECTOR+2)] = temp.s2;
            lmem[(ly)*loff + (lx*VECTOR+3)] = temp.s3;
            lmem[(ly)*loff + (lx*VECTOR+4)] = temp.s4;
            lmem[(ly)*loff + (lx*VECTOR+5)] = temp.s5;
            lmem[(ly)*loff + (lx*VECTOR+6)] = temp.s6;
            lmem[(ly)*loff + (lx*VECTOR+7)] = temp.s7;
          #endif
        }
      }
    }
  }
}
#endif

// Loads data (plus the halos) into local memory
#if LOCAL == 1
inline void LoadLocalPlusHalo(__local float *lmem, const int loff,
                              const __global float* src, const int goff,
                              const int gid_x, const int gid_y, const int lid_x, const int lid_y) {

  // Loop over the amount of work per thread
  #pragma unroll
  for (int wx=0; wx<WPTX; ++wx) {
    const int lx = lid_x*WPTX + wx;
    const int gx = gid_x*WPTX + wx;
    #pragma unroll
    for (int wy=0; wy<WPTY; ++wy) {
      const int ly = lid_y*WPTY + wy;
      const int gy = gid_y*WPTY + wy;

      // Computes the conditionals
      const bool xst = lx < HFS;
      const bool xlt = lx >= TBX*WPTX-HFS;
      const bool yst = ly < HFS;
      const bool ylt = ly >= TBY*WPTY-HFS;

      // In the centre
                        lmem[(ly+1*HFS)*loff + (lx+1*HFS)] = src[(gy+1*HFS)*goff + (gx+1*HFS)];
      // On the x-border
      if (xst       ) { lmem[(ly+1*HFS)*loff + (lx      )] = src[(gy+1*HFS)*goff + (gx      )]; }
      if (xlt       ) { lmem[(ly+1*HFS)*loff + (lx+2*HFS)] = src[(gy+1*HFS)*goff + (gx+2*HFS)]; }
      // On the y-border
      if (yst       ) { lmem[(ly      )*loff + (lx+1*HFS)] = src[(gy      )*goff + (gx+1*HFS)]; }
      if (ylt       ) { lmem[(ly+2*HFS)*loff + (lx+1*HFS)] = src[(gy+2*HFS)*goff + (gx+1*HFS)]; }
      // On both the x and y borders
      if (xst && yst) { lmem[(ly      )*loff + (lx      )] = src[(gy      )*goff + (gx      )]; }
      if (xst && ylt) { lmem[(ly+2*HFS)*loff + (lx      )] = src[(gy+2*HFS)*goff + (gx      )]; }
      if (xlt && yst) { lmem[(ly      )*loff + (lx+2*HFS)] = src[(gy      )*goff + (gx+2*HFS)]; }
      if (xlt && ylt) { lmem[(ly+2*HFS)*loff + (lx+2*HFS)] = src[(gy+2*HFS)*goff + (gx+2*HFS)]; }
    }
  }
}
#endif

// =================================================================================================

// Accumulates in the local memory
#if LOCAL == 1 || LOCAL == 2
inline void AccumulateLocal(__local float *lmem, const int loff,
                            __constant float* coeff, float acc[WPTY][WPTX],
                            const int lid_x, const int lid_y) {

  // Caches data from local memory into registers
  float rmem[FS+WPTY-1][FS+WPTX-1];
  #pragma unroll
  for (int x=0; x<FS+(WPTX-1); ++x) {
    const int lx = lid_x*WPTX + x;
    #pragma unroll
    for (int y=0; y<FS+(WPTY-1); ++y) {
      const int ly = lid_y*WPTY + y;
      rmem[y][x] = lmem[ly*loff + lx];
    }
  }

  // Loops over the neighbourhood
  #pragma unroll UNROLL_FACTOR
  for (int fx=0; fx<FS; ++fx) {
    #pragma unroll UNROLL_FACTOR
    for (int fy=0; fy<FS; ++fy) {
      const float coefficient = coeff[fy*FS + fx];

      // Performs the accumulation
      #pragma unroll
      for (int wx=0; wx<WPTX; ++wx) {
        #pragma unroll
        for (int wy=0; wy<WPTY; ++wy) {
          acc[wy][wx] += coefficient * rmem[wy+fy][wx+fx];
        }
      }
    }
  }
}
#endif

// Accumulates in the global memory
#if LOCAL == 0
inline void AccumulateGlobal(const __global float* src, const int goff,
                             __constant float* coeff, float acc[WPTY][WPTX],
                             const int gid_x, const int gid_y) {

  // Caches data from global memory into registers
  float rmem[FS+WPTY-1][FS+WPTX-1];
  #pragma unroll
  for (int x=0; x<FS+(WPTX-1); ++x) {
    const int gx = gid_x*WPTX + x;
    #pragma unroll
    for (int y=0; y<FS+(WPTY-1); ++y) {
      const int gy = gid_y*WPTY + y;
      rmem[y][x] = src[gy*goff + gx];
    }
  }

  // Loops over the neighbourhood
  #pragma unroll UNROLL_FACTOR
  for (int fx=0; fx<FS; ++fx) {
    #pragma unroll UNROLL_FACTOR
    for (int fy=0; fy<FS; ++fy) {
      const float coefficient = coeff[fy*FS + fx];

      // Performs the accumulation
      #pragma unroll
      for (int wx=0; wx<WPTX; ++wx) {
        #pragma unroll
        for (int wy=0; wy<WPTY; ++wy) {
          acc[wy][wx] += coefficient * rmem[wy+fy][wx+fx];
        }
      }
    }
  }
}
#endif

// =================================================================================================

// Stores the result into global memory
inline void StoreResult(__global floatvec* dest, const int goff, float acc[WPTY][WPTX],
                        const int gid_x, const int gid_y) {
  #pragma unroll
  for (int wx=0; wx<WPTX/VECTOR; ++wx) {
    const int gx = gid_x*WPTX/VECTOR + wx;
    #pragma unroll
    for (int wy=0; wy<WPTY; ++wy) {
      const int gy = gid_y*WPTY + wy;
      floatvec temp;
      #if VECTOR == 1
        temp = acc[wy][wx*VECTOR];
      #elif VECTOR == 2
        temp.x = acc[wy][wx*VECTOR  ];
        temp.y = acc[wy][wx*VECTOR+1];
      #elif VECTOR == 4
        temp.x = acc[wy][wx*VECTOR  ];
        temp.y = acc[wy][wx*VECTOR+1];
        temp.z = acc[wy][wx*VECTOR+2];
        temp.w = acc[wy][wx*VECTOR+3];
      #elif VECTOR == 8
        temp.s0 = acc[wy][wx*VECTOR  ];
        temp.s1 = acc[wy][wx*VECTOR+1];
        temp.s2 = acc[wy][wx*VECTOR+2];
        temp.s3 = acc[wy][wx*VECTOR+3];
        temp.s4 = acc[wy][wx*VECTOR+4];
        temp.s5 = acc[wy][wx*VECTOR+5];
        temp.s6 = acc[wy][wx*VECTOR+6];
        temp.s7 = acc[wy][wx*VECTOR+7];
      #endif
      dest[gy*goff/VECTOR + gx] = temp;
    }
  }
}

// =================================================================================================

// Tuneable implementation of the 2D convolution example
#if LOCAL == 0
__attribute__((reqd_work_group_size(TBX, TBY, 1)))
__kernel void conv(const int goff, const int dummy,
                   const __global float* src,
                   __constant float* coeff,
                   __global floatvec* dest) {

  // Thread identifiers
  const int gid_x = get_global_id(0); // From 0 to goff/WPTX-1
  const int gid_y = get_global_id(1); // From 0 to dummy/WPTY-1

  // Initializes the accumulation registers
  float acc[WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in global memory
  AccumulateGlobal(src, goff+2*HFS, coeff, acc, gid_x, gid_y);

  // Computes and stores the result
  StoreResult(dest, goff, acc, gid_x, gid_y);
}
#endif

// =================================================================================================

// Tuneable implementation of the 2D convolution example
#if LOCAL == 1
__attribute__((reqd_work_group_size(TBX, TBY, 1)))
__kernel void conv(const int goff, const int dummy,
                   const __global float* src,
                   __constant float* coeff,
                   __global floatvec* dest) {

  // Thread identifiers
  const int gid_x = get_global_id(0); // From 0 to goff/WPTX-1
  const int gid_y = get_global_id(1); // From 0 to dummy/WPTY-1

  // Local memory
  const int lid_x = get_local_id(0); // From 0 to TBX
  const int lid_y = get_local_id(1); // From 0 to TBY
  __local float lmem[(TBY*WPTY + 2*HFS) * (TBX*WPTX + 2*HFS + PADDING)];
  const int loff = TBX*WPTX + 2*HFS + PADDING;

  // Caches data into local memory
  LoadLocalPlusHalo(lmem, loff, src, goff+2*HFS, gid_x, gid_y, lid_x, lid_y);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Initializes the accumulation registers
  float acc[WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in local memory
  AccumulateLocal(lmem, loff, coeff, acc, lid_x, lid_y);

  // Computes and stores the result
  StoreResult(dest, goff, acc, gid_x, gid_y);
}
#endif

// =================================================================================================

// Tuneable implementation of the 2D convolution example
#if LOCAL == 2
__attribute__((reqd_work_group_size(TBX_XL, TBY_XL, 1)))
__kernel void conv(const int goff, const int dummy,
                   const __global floatvec* src,
                   __constant float* coeff,
                   __global floatvec* dest) {

  // Thread identifiers
  const int gid_x = get_local_id(0) + TBX*get_group_id(0);
  const int gid_y = get_local_id(1) + TBY*get_group_id(1);

  // Local memory
  const int lid_x = get_local_id(0); // From 0 to (TBX + 2*HFS)
  const int lid_y = get_local_id(1); // From 0 to (TBY + 2*HFS)
  __local float lmem[(TBY*WPTY + 2*HFS) * (TBX*WPTX + 2*HFS + PADDING)];
  const int loff = TBX*WPTX + 2*HFS + PADDING;

  // Caches data into local memory
  LoadLocalFull(lmem, loff, src, goff+2*HFS, gid_x, gid_y, lid_x, lid_y);

  // Synchronizes all threads in a workgroup
  barrier(CLK_LOCAL_MEM_FENCE);

  // Cancels some threads (those that were only used for loading halo data)
  if ((lid_x >= TBX) || (lid_y >= TBY)) {
    return;
  }

  // Initializes the accumulation registers
  float acc[WPTY][WPTX];
  InitAccRegisters(acc);

  // Accumulates in local memory
  AccumulateLocal(lmem, loff, coeff, acc, lid_x, lid_y);

  // Computes and stores the result
  StoreResult(dest, goff, acc, gid_x, gid_y);
}
#endif

// =================================================================================================
