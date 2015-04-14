
// =================================================================================================
// This file is part of the CLTune project, which loosely follows the Google C++ styleguide and uses
// a tab-size of two spaces and a max-width of 100 characters per line.
//
// Author: cedric.nugteren@surfsara.nl (Cedric Nugteren)
//
// This file contains an example OpenCL kernel as part of the gemm.cc example. It is an optimized
// matrix-multiplication kernel according to the paper by Matsumoto et al. and the tutorial on
// http://www.cedricnugteren.nl/tutorial.php. It is fully configurable (and tunable!) using more or
// less the same parameters/naming conventions as in the paper. It supports single and double
// precision (SGEMM/DGEMM) through a pre-processor define.
//
// Note: this kernel requires a compiler compliant to OpenCL 1.1 or higher.
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
//
// Matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// Parameters determined by the tuner
// MWG       : Tile-size in dimension M (e.g. 64, 128)
// NWG       : Tile-size in dimension N (e.g. 64, 128)
// KWG       : Tile-size in dimension K (e.g. 8, 16)
// MDIMC     : Threads per workgroup in M-dimension (e.g. 8, 16, 32)
// NDIMC     : Threads per workgroup in N-dimension (e.g. 8, 16, 32)
// MDIMA     : Re-shaped tile dimension of matrix A: KDIMA * MDIMA
// NDIMB     : Re-shaped tile dimension of matrix B: KDIMB * NDIMB
// KWI       : Unroll factor of the KWG loop (smaller or equal than KWG)
// VWM       : Vector width of matrices A and C (supported 1, 2, 4, and 8)
// VWN       : Vector width of matrix B (supported 1, 2, 4, and 8)
// STRM      : Use strided access within a thread in the M-dimension (1) or not (0)
// STRN      : Use strided access within a thread in the N-dimension (1) or not (0)
// SA        : Use local/shared memory to cache matrix A (1) or not (0)
// SB        : Use local/shared memory to cache matrix B (1) or not (0)
// PRECISION : Whether to use single (32) or double (64) precision data-types
//
// =================================================================================================

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#define USE_MAD 0                     // Uses the non IEEE-754 compliant mad() function

// =================================================================================================

// Data-type: single or double precision
#if PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  #define ZERO 0.0f
#elif PRECISION == 64
  #if __OPENCL_VERSION__ <= CL_VERSION_1_1 // This the default on OpenCL 1.2 or higher
     #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  #define ZERO 0.0
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#endif

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
inline void GlobalToLocalA(const __global realM* restrict agm, __local realM* alm,
                           const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int mia=0; mia<MWA/VWM; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWA; ++kia) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = kia + la1*KWA;
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void GlobalToLocalB(const __global realN* restrict bgm, __local realN* blm,
                           const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int kib=0; kib<KWB; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWB/VWN; ++nib) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = kib + lb1*KWB;
      int idn = ng + get_group_id(1)*(NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0
inline void GlobalToPrivateA(const __global realM* restrict agm, realM apm[MWI/VWM],
                             const int kSizeM, const int idk, const int kwg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {

    // Computes the indices based on strided/non-strided access
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif

    // Computes the indices for the global memory
    int idm = mg + get_group_id(0)*(MWG/VWM);

    // Loads the data from global memory (not transposed) and stores into registers
    apm[mi] = agm[idk*(kSizeM/VWM) + idm];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0
inline void GlobalToPrivateB(const __global realN* restrict bgm, realN bpm[NWI/VWN],
                             const int kSizeN, const int idk) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {

    // Computes the indices based on strided/non-strided access
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif

    // Computes the indices for the global memory
    int idn = ng + get_group_id(1)*(NWG/VWN);

    // Loads the data from global memory (transposed) and stores into registers
    bpm[ni] = bgm[idk*(kSizeN/VWN) + idn];
  }
}
#endif

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
inline void LocalToPrivateA(__local realM* alm, realM apm[MWI/VWM], const int kg) {
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #if STRM == 0
      int mg = mi + get_local_id(0)*(MWI/VWM);
    #elif STRM == 1
      int mg = get_local_id(0) + mi*MDIMC;
    #endif
    apm[mi] = alm[kg*(MWG/VWM) + mg];
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
inline void LocalToPrivateB(__local realN* blm, realN bpm[NWI/VWN], const int kg) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #if STRN == 0
      int ng = ni + get_local_id(1)*(NWI/VWN);
    #elif STRN == 1
      int ng = get_local_id(1) + ni*NDIMC;
    #endif
    bpm[ni] = blm[kg*(NWG/VWN) + ng];
  }
}
#endif

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm
inline void StoreResults(__global realM* cgm, realM cpm[NWI][MWI/VWM], const int kSizeM) {
  #pragma unroll
  for (int ni=0; ni<NWI; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if STRM == 0
        int mg = mi + get_local_id(0)*(MWI/VWM);
      #elif STRM == 1
        int mg = get_local_id(0) + mi*MDIMC;
      #endif
      #if STRN == 0
        int ng = ni + get_local_id(1)*NWI;
      #elif STRN == 1
        int ng = ni%VWN + get_local_id(1)*VWN + (ni/VWN)*VWN*NDIMC;
      #endif
      int idm = mg + get_group_id(0)*(MWG/VWM);
      int idn = ng + get_group_id(1)*NWG;
      int index = idn*(kSizeM/VWM) + idm;
      cgm[index] = cpm[ni][mi];
    }
  }
}

// =================================================================================================

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulate(realM cpm[NWI][MWI/VWM], realM apm[MWI/VWM], realN bpm[NWI/VWN]) {
  #pragma unroll
  for (int ni=0; ni<NWI/VWN; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWI/VWM; ++mi) {
      #if USE_MAD == 1
        #if VWN == 1
          cpm[ni*VWN + 0][mi] = mad(apm[mi], bpm[ni], cpm[ni*VWN + 0][mi]);
        #elif VWN == 2
          cpm[ni*VWN + 0][mi] = mad(apm[mi], bpm[ni].x, cpm[ni*VWN + 0][mi]);
          cpm[ni*VWN + 1][mi] = mad(apm[mi], bpm[ni].y, cpm[ni*VWN + 1][mi]);
        #elif VWN == 4
          cpm[ni*VWN + 0][mi] = mad(apm[mi], bpm[ni].x, cpm[ni*VWN + 0][mi]);
          cpm[ni*VWN + 1][mi] = mad(apm[mi], bpm[ni].y, cpm[ni*VWN + 1][mi]);
          cpm[ni*VWN + 2][mi] = mad(apm[mi], bpm[ni].z, cpm[ni*VWN + 2][mi]);
          cpm[ni*VWN + 3][mi] = mad(apm[mi], bpm[ni].w, cpm[ni*VWN + 3][mi]);
        #elif VWN == 8
          cpm[ni*VWN + 0][mi] = mad(apm[mi], bpm[ni].s0, cpm[ni*VWN + 0][mi]);
          cpm[ni*VWN + 1][mi] = mad(apm[mi], bpm[ni].s1, cpm[ni*VWN + 1][mi]);
          cpm[ni*VWN + 2][mi] = mad(apm[mi], bpm[ni].s2, cpm[ni*VWN + 2][mi]);
          cpm[ni*VWN + 3][mi] = mad(apm[mi], bpm[ni].s3, cpm[ni*VWN + 3][mi]);
          cpm[ni*VWN + 4][mi] = mad(apm[mi], bpm[ni].s4, cpm[ni*VWN + 4][mi]);
          cpm[ni*VWN + 5][mi] = mad(apm[mi], bpm[ni].s5, cpm[ni*VWN + 5][mi]);
          cpm[ni*VWN + 6][mi] = mad(apm[mi], bpm[ni].s6, cpm[ni*VWN + 6][mi]);
          cpm[ni*VWN + 7][mi] = mad(apm[mi], bpm[ni].s7, cpm[ni*VWN + 7][mi]);
        #endif
      #else
        #if VWN == 1
          cpm[ni*VWN + 0][mi] += apm[mi] * bpm[ni];
        #elif VWN == 2
          cpm[ni*VWN + 0][mi] += apm[mi] * bpm[ni].x;
          cpm[ni*VWN + 1][mi] += apm[mi] * bpm[ni].y;
        #elif VWN == 4
          cpm[ni*VWN + 0][mi] += apm[mi] * bpm[ni].x;
          cpm[ni*VWN + 1][mi] += apm[mi] * bpm[ni].y;
          cpm[ni*VWN + 2][mi] += apm[mi] * bpm[ni].z;
          cpm[ni*VWN + 3][mi] += apm[mi] * bpm[ni].w;
        #elif VWN == 8
          cpm[ni*VWN + 0][mi] += apm[mi] * bpm[ni].s0;
          cpm[ni*VWN + 1][mi] += apm[mi] * bpm[ni].s1;
          cpm[ni*VWN + 2][mi] += apm[mi] * bpm[ni].s2;
          cpm[ni*VWN + 3][mi] += apm[mi] * bpm[ni].s3;
          cpm[ni*VWN + 4][mi] += apm[mi] * bpm[ni].s4;
          cpm[ni*VWN + 5][mi] += apm[mi] * bpm[ni].s5;
          cpm[ni*VWN + 6][mi] += apm[mi] * bpm[ni].s6;
          cpm[ni*VWN + 7][mi] += apm[mi] * bpm[ni].s7;
        #endif
      #endif
    }
  }
}

// =================================================================================================

// Main entry of the kernel. This function contains the basic skeleton, the functionality is
// provided by the inlined functions above.
__attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
__kernel void gemm_fast(const int kSizeM, const int kSizeN, const int kSizeK,
                        const __global realM* restrict agm,
                        const __global realN* restrict bgm,
                        __global realM* cgm) {

  // Combined thread identifier
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif
  
  // Allocates workitem-private memory (registers)
  realM apm[MWI/VWM];
  realN bpm[NWI/VWN];
  realM cpm[NWI][MWI/VWM];

  // Initializes the accumulation registers
  #pragma unroll
  for (int mi=0; mi<MWI/VWM; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWI; ++ni) {
      cpm[ni][mi] = (realM)ZERO;
    }
  }

  // Loops over all workgroup tiles
  for (int kwg=0; kwg<kSizeK; kwg+=KWG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif

    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi=0; pwi<KWG; pwi+=KWI) {
      #pragma unroll
      for (int pit=0; pit<KWI; ++pit) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi+pit;
        #endif

        // Loads data: local --> private (matrix A)
        #if SA == 1
          LocalToPrivateA(alm, apm, kg);
        // Loads data: off-chip --> private (matrix A)
        #else
          GlobalToPrivateA(agm, apm, kSizeM, idk, kwg);
        #endif

        // Loads data: local --> private (matrix B)
        #if SB == 1
          LocalToPrivateB(blm, bpm, kg);
        // Loads data: off-chip --> private (matrix B)
        #else
          GlobalToPrivateB(bgm, bpm, kSizeN, idk);
        #endif

        // Performs the accumulation (Cpm += Apm * Bpm)
        MultiplyAccumulate(cpm, apm, bpm);
      }
    }

    // Synchronizes all threads in a workgroup
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }

  // Stores an MWG * NWG tile of results
  StoreResults(cgm, cpm, kSizeM);
}

// =================================================================================================
