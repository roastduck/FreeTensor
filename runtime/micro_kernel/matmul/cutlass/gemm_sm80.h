/**
 * This file is borrowed from
 * https://github.com/nox-410/tvm.tl/blob/tl/src/tl/tl_templates/gemm_sm80.h
 * under Apache Lincense, and modified for use.
 */

#ifndef MICRO_KERNEL_MATMUL_CUTLASS_GEMM_SM80_H
#define MICRO_KERNEL_MATMUL_CUTLASS_GEMM_SM80_H

#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/numeric_types.h>

using cutlass::gemm::GemmShape;

template <typename A_type, typename B_type, typename C_type>
struct DispatchInstruction;

template <>
struct DispatchInstruction<cutlass::half_t, cutlass::half_t, cutlass::half_t> {
    using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<cutlass::half_t, cutlass::half_t, float> {
    using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
    using Shape = GemmShape<16, 8, 16>;
};
template <>
struct DispatchInstruction<cutlass::tfloat32_t, cutlass::tfloat32_t, float> {
    using Shape = GemmShape<16, 8, 8>;
};
template <> struct DispatchInstruction<double, double, double> {
    using Shape = GemmShape<8, 8, 4>;
};
template <> struct DispatchInstruction<int8_t, int8_t, int> {
    using Shape = GemmShape<16, 8, 32>;
};

template <bool transpose> struct DispatchSharedMemoryLayout;

template <> struct DispatchSharedMemoryLayout<true> {
    using Layout = cutlass::layout::ColumnMajor;
};
template <> struct DispatchSharedMemoryLayout<false> {
    using Layout = cutlass::layout::RowMajor;
};

template <typename Shape, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
class GemmTensorOp {
  public:
    using A_type =
        typename std::conditional<std::is_same<A_type_raw, float>::value,
                                  cutlass::tfloat32_t, A_type_raw>::type;
    using B_type =
        typename std::conditional<std::is_same<B_type_raw, float>::value,
                                  cutlass::tfloat32_t, A_type_raw>::type;
    using C_type = C_type_raw;
    using InstructionShape =
        typename DispatchInstruction<A_type, B_type, C_type>::Shape;
    using SMemLayoutA = typename DispatchSharedMemoryLayout<trans_A>::Layout;
    using SMemLayoutB = typename DispatchSharedMemoryLayout<trans_B>::Layout;

    using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
        cutlass::arch::Mma<
            InstructionShape, 32, A_type, cutlass::layout::RowMajor, B_type,
            cutlass::layout::ColumnMajor, C_type, cutlass::layout::RowMajor,
            cutlass::arch::OpMultiplyAdd>,
        cutlass::MatrixShape<1, 1>>;

    static_assert(Shape::kM % num_warp_m == 0);
    static_assert(Shape::kN % num_warp_n == 0);

    using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
        GemmShape<Shape::kM / num_warp_m, Shape::kN / num_warp_n,
                  InstructionShape::kK>,
        A_type, SMemLayoutA, B_type, SMemLayoutB, C_type,
        cutlass::layout::RowMajor, Policy, 1,
        true /* accumulate in row major */>;

    using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
    using TensorRefB = typename MmaWarp::IteratorB::TensorRef;
    using FragmentA = typename MmaWarp::FragmentA;
    using FragmentB = typename MmaWarp::FragmentB;
    using FragmentC = typename MmaWarp::FragmentC;
    using IteratorA = typename MmaWarp::IteratorA;
    using IteratorB = typename MmaWarp::IteratorB;

    static_assert(Shape::kK % InstructionShape::kK == 0);
    static int constexpr kKgroups = Shape::kK / InstructionShape::kK;

    static CUTLASS_DEVICE void body(const A_type_raw *pA, const B_type_raw *pB,
                                    FragmentC &accum, int lda, int ldb,
                                    double alpha, double beta,
                                    const int warp_idx_m, const int warp_idx_n,
                                    const int lane_id) {
        MmaWarp mma_op;
        FragmentA frag_A;
        FragmentB frag_B;
        const TensorRefA ref_A((A_type *)pA, lda);
        const TensorRefB ref_B((B_type *)pB, ldb);
        IteratorA iter_A(ref_A, lane_id);
        IteratorB iter_B(ref_B, lane_id);
        iter_A.add_tile_offset({warp_idx_m, 0});
        iter_B.add_tile_offset({0, warp_idx_n});

        // TODO: Check all cases of alpha and beta
        // TODO: Static checking of alpha and beta
        if (beta == 0) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentC::kElements; i++) {
                accum[i] = 0;
            }
        } else {
            assert(beta == 1);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kKgroups; ++k) {
            iter_A.load(frag_A);
            iter_B.load(frag_B);
            ++iter_A;
            ++iter_B;
            mma_op(accum, frag_A, frag_B, accum);
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
    using MMA = GemmTensorOp<GemmShape<M, N, K>, num_warp_m, num_warp_n,
                             trans_A, trans_B, A_type, B_type, C_type>;
    using FragmentC = typename MMA::FragmentC;
    MMA::body(pA + warp_id_batch * stridea, pB + warp_id_batch * strideb,
              *(FragmentC *)(accum /* no thread offset */), lda, ldb, alpha,
              beta, warp_id_m, warp_id_n, lane_id);
}

#endif // MICRO_KERNEL_MATMUL_CUTLASS_GEMM_SM80_H
