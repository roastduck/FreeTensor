/**
 * This file is borrowed from
 * https://github.com/nox-410/tvm.tl/blob/tl/src/tl/tl_templates/cute_gemm.h
 * under Apache Lincense, and modified for use.
 */

#pragma once

#include <cute/algorithm/copy.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

template <> struct DispatchInstruction<half_t, half_t, half_t> {
    using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
};
template <> struct DispatchInstruction<double, double, double> {
    using MMA = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
};

template <int Bits, int N, int K, bool K_inner, typename Enable = void>
struct OperandTraits {
    static constexpr int stride = K_inner ? K : N;
    using Layout = typename std::conditional<
        K_inner, Layout<Shape<Int<N>, Int<K>>, Shape<Int<K>, _1>>,
        Layout<Shape<Int<N>, Int<K>>, Shape<_1, Int<N>>>>::type;
    using Copy = DefaultCopy;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
class GemmTensorOp {
  public:
    using A_type =
        typename std::conditional<std::is_same<A_type_raw, float>::value,
                                  tfloat32_t, A_type_raw>::type;
    using B_type =
        typename std::conditional<std::is_same<B_type_raw, float>::value,
                                  tfloat32_t, A_type_raw>::type;
    using C_type = C_type_raw;

    using Instruction = DispatchInstruction<A_type, B_type, C_type>;

    using OperandATraits =
        OperandTraits<sizeof_bits<A_type>::value, M, K, !trans_A>;
    using OperandBTraits =
        OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B>;
    using SmemLayoutA = typename OperandATraits::Layout;
    using SmemLayoutB = typename OperandBTraits::Layout;
    using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
    using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

    using TileMma = TiledMMA<typename Instruction::MMA,
                             Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>
                             /*,typename Instruction::MMA_Group*/>;

    static CUTE_DEVICE void body(const A_type_raw *pA, const B_type_raw *pB,
                                 C_type_raw *pC, int lda, int ldb, double alpha,
                                 double beta, int warp_id_m, int warp_id_n,
                                 int lane_id) {

        int tid = (warp_id_n * num_warp_m + warp_id_m) * 32 + lane_id;
        // change the layout!!!
        Tensor sA = make_tensor(make_smem_ptr((A_type *)(pA)), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr((B_type *)(pB)), SmemLayoutB{});
        TileMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);
        auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
        auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
        auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
        auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

        Tensor tCrA = thr_mma.partition_fragment_A(sA);
        Tensor tCrB = thr_mma.partition_fragment_B(sB);
        Tensor tCsA = thr_copy_A.partition_S(sA);
        Tensor tCsB = thr_copy_B.partition_S(sB);

        Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
        Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

        Tensor acc =
            make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                        partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

        int num_tile_k = size<2>(tCrA);
        CUTE_UNROLL
        for (int k = 0; k < num_tile_k; ++k) {
            copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
            copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
            gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
        }
    }
};

template <int M, int N, int K, int num_warp_batch, int num_warp_m,
          int num_warp_n, bool trans_A, bool trans_B, typename A_type,
          typename B_type, typename C_type>
CUTLASS_DEVICE void matmul_thread(const A_type *pA, const B_type *pB,
                                  C_type *accum, int lda, int ldb, int stridea,
                                  int strideb, int stridec, double alpha,
                                  double beta, int warp_id_batch, int warp_id_m,
                                  int warp_id_n, int lane_id) {
    using MMA = GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A, trans_B,
                             A_type, B_type, C_type>;
    MMA::body(pA + warp_id_batch * stridea, pB + warp_id_batch * strideb,
              (accum /* no thread offset */), lda, ldb, alpha, beta, warp_id_m,
              warp_id_n, lane_id);
}
