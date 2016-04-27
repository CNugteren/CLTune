
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an example CUDA kernel as part of the conv_simple.cc example. This assumes
// the workgroup dimension is a multiple of the matrix sizes.
//
// =================================================================================================

// Settings (synchronise these among all "conv_simple.*" files)
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

// Vector data-types
#if VECTOR == 1
    typedef float floatvec;
#elif VECTOR == 2
    typedef float2 floatvec;
#elif VECTOR == 4
    typedef float4 floatvec;
#endif

// =================================================================================================

// Tuneable implementation of the 2D convolution example
extern "C" __global__ void conv(const int sizeX, const int sizeY,
                                const float* src,
                                float* coeff,
                                floatvec* dest) {

  // Thread identifiers
  const int gid_x = blockIdx.x * TBX + threadIdx.x; // From 0 to sizeX/WPTX-1
  const int gid_y = blockIdx.y * TBY + threadIdx.y; // From 0 to sizeY/WPTY-1

  // Initializes the accumulation registers
  float acc[WPTY][WPTX];
  #pragma unroll
  for (int wx=0; wx<WPTX; ++wx) {
    #pragma unroll
    for (int wy=0; wy<WPTY; ++wy) {
      acc[wy][wx] = 0.0f;
    }
  }

  // Caches data from global memory into registers
  float rmem[FS+WPTY-1][FS+WPTX-1];
  #pragma unroll
  for (int x=0; x<FS+(WPTX-1); ++x) {
    const int gx = gid_x*WPTX + x;
    #pragma unroll
    for (int y=0; y<FS+(WPTY-1); ++y) {
      const int gy = gid_y*WPTY + y;
      rmem[y][x] = src[gy*sizeX + gx];
    }
  }

  // Loops over the neighbourhood
  for (int fx=0; fx<FS; ++fx) {
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

  // Computes and stores the result
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
      dest[gy*sizeX/VECTOR + gx] = temp;
    }
  }
}


// =================================================================================================
