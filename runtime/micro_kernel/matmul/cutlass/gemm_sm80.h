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

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
template <> struct DispatchInstruction<half_t, half_t, half_t> {
    using MMA = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <> struct DispatchInstruction<half_t, half_t, float> {
    using MMA = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <> struct DispatchInstruction<bfloat16_t, bfloat16_t, float> {
    using MMA = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <> struct DispatchInstruction<tfloat32_t, tfloat32_t, float> {
    using MMA = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <> struct DispatchInstruction<int8_t, int8_t, int> {
    using MMA = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _1>>;
};
template <> struct DispatchInstruction<double, double, double> {
    using MMA = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
    using MMA_Group = Layout<Shape<_2, _2, _1>>;
};
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
template <> struct DispatchInstruction<half_t, half_t, float> {
    using MMA = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using MMA_Group = Layout<Shape<_1, _2, _2>>;
};
#endif

template <int Bits, int N, int K, bool K_inner, typename Enable = void>
struct OperandTraits {
    // Primary template, use padded layout and default copy
    static constexpr int stride = K_inner ? K : N;
    static constexpr int padded =
        stride % (256 / Bits) == 0 ? stride + 128 / Bits : stride;
    using Layout = typename std::conditional<
        K_inner, Layout<Shape<Int<N>, Int<K>>, Shape<Int<padded>, _1>>,
        Layout<Shape<Int<N>, Int<K>>, Shape<_1, Int<padded>>>>::type;
    using Copy = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<16, N, K, true,
                     typename std::enable_if<K % 64 == 32>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, true,
                     typename std::enable_if<K % 64 == 0>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<16, N, K, false,
                     typename std::enable_if<N % 64 == 32>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
    using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                          Step<_2, _1>{}));
    using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<16, N, K, false,
                     typename std::enable_if<N % 64 == 0>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
    using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                          Step<_2, _1>{}));
    using Copy = SM75_U16x8_LDSM_T;
};

template <int N, int K>
struct OperandTraits<32, N, K, true,
                     typename std::enable_if<K % 32 == 0>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, true,
                     typename std::enable_if<K % 32 == 16>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<2, 2, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<32, N, K, false,
                     typename std::enable_if<N % 32 == 0>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
    using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                          Step<_2, _1>{}));
    using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<32, N, K, false,
                     typename std::enable_if<N % 32 == 16>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<2, 2, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
    using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{},
                                          Step<_2, _1>{}));
    using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K>
struct OperandTraits<8, N, K, true,
                     typename std::enable_if<K % 128 == 64>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<8, N, K, true,
                     typename std::enable_if<K % 128 == 0>::type> {
    using LayoutAtom = decltype(composition(
        Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
    using Layout =
        decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<K>>{}));
    using Copy = SM75_U32x4_LDSM_N;
};

template <int N, int K>
struct OperandTraits<64, N, K, true,
                     typename std::enable_if<K % 16 == 0>::type> {
    // using LayoutAtom =
    //     decltype(composition(Swizzle<2, 0, 4>{}, Layout<Shape<_4, _16>,
    //     Stride<_16, _1>>{}));
    // using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>,
    // Int<K>>{}));
    using Layout = Layout<Shape<Int<N>, Int<K>>, Stride<Int<K>, _1>>;
    using Copy = DefaultCopy;
};

template <int N, int K>
struct OperandTraits<64, N, K, false,
                     typename std::enable_if<N % 16 == 0>::type> {
    // using LayoutAtom =
    //     decltype(composition(Swizzle<2, 2, 2>{}, Layout<Shape<_16, _4>,
    //     Stride<_1, _16>>{}));
    // using Layout = decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>,
    // Int<K>>{}, Step<_2, _1>{}));
    using Layout = Layout<Shape<Int<N>, Int<K>>, Stride<_1, Int<N>>>;
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

    using TileMma =
        TiledMMA<typename Instruction::MMA,
                 Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>>;

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
