#ifndef MICRO_KERNEL_MATMUL_CUTLASS_GEMM_H
#define MICRO_KERNEL_MATMUL_CUTLASS_GEMM_H

#if (defined(__CUDA_ARCH__)) // Device code

#if (__CUDA_ARCH__ >= 800)
#include "gemm_sm80.h"
#else
#error "Unsupported architecture"
#endif

#else // Host code

// Only declaration is needed
template <int M, int N, int K, int num_warp_batch, int num_warp_m,
          int num_warp_n, bool trans_A, bool trans_B, typename A_type,
          typename B_type, typename C_type>
__device__ void matmul_thread(const A_type *pA, const B_type *pB, C_type *accum,
                              int lda, int ldb, int stridea, int strideb,
                              int stridec, int warp_id_batch, int warp_id_m,
                              int warp_id_n, int lane_id);

#endif

#endif // MICRO_KERNEL_MATMUL_CUTLASS_GEMM_H
