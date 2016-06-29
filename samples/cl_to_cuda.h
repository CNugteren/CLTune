
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains an (incomplete) header to interpret OpenCL kernels as CUDA kernels.
//
// =================================================================================================

// Replaces the OpenCL keywords with CUDA equivalent
#define __kernel __placeholder__
#define __global 
#define __placeholder__ extern "C" __global__
#define __local __shared__
#define restrict __restrict__
#define __constant const
#define inline __device__ inline // assumes all device functions are annotated with inline in OpenCL

// Replaces OpenCL synchronisation with CUDA synchronisation
#define barrier(x) __syncthreads()

// Replaces the OpenCL get_xxx_ID with CUDA equivalents
__device__ int get_local_id(int x) {
    return (x == 0) ? threadIdx.x : threadIdx.y;
}
__device__ int get_group_id(int x) {
    return (x == 0) ? blockIdx.x : blockIdx.y;
}
__device__ int get_global_id(int x) {
    return (x == 0) ? blockIdx.x*blockDim.x + threadIdx.x : blockIdx.y*blockDim.y + threadIdx.y;
}

// Adds the data-types which are not available natively under CUDA
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7; } float8;
typedef struct { float s0; float s1; float s2; float s3;
                 float s4; float s5; float s6; float s7;
                 float s8; float s9; float s10; float s11;
                 float s12; float s13; float s14; float s15; } float16;

// =================================================================================================
