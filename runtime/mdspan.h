#ifndef FREE_TENSOR_MDSPAN_H
#define FREE_TENSOR_MDSPAN_H

#include <cstdint>
#include <cstdlib>

// CUDA supports only `printf` in kernel. Neither `fprintf` nor `iostream` is
// supported. If unsure whether we are in a CUDA kernel, always use `printf`
#include <cstdio>

#include "../3rd-party/mdspan/mdspan.hpp"

#ifdef __CUDA_ARCH__
#define FUNC_ATTR __attribute__((always_inline)) __host__ __device__
#else
#define FUNC_ATTR __attribute__((always_inline))
#endif

namespace stdex = std::experimental;

using stdex::dynamic_extent;
using stdex::mdspan;

template <size_t... S> using extents = stdex::extents<size_t, S...>;

// mdspan with runtime range check for debugging

template <typename ElementType, typename Extents>
class mdspan_dbg : public stdex::mdspan<ElementType, Extents> {
    typedef stdex::mdspan<ElementType, Extents> BaseClass;

    template <size_t DIM, typename FirstIdx, typename... OtherIdx>
    FUNC_ATTR bool checkDims(FirstIdx &&first, OtherIdx &&...others) const {
        if (first < 0 || (int64_t)first >= (int64_t)this->extent(DIM)) {
            return false;
        }
        return checkDims<DIM + 1>(std::forward<OtherIdx>(others)...);
    }
    template <size_t DIM> FUNC_ATTR bool checkDims() const { return true; }

    template <size_t DIM, typename FirstIdx, typename... OtherIdx>
    FUNC_ATTR void printIndices(FirstIdx &&first, OtherIdx &&...others) const {
        printf(DIM > 0 ? ", %lld" : "%lld", (long long)first);
        printIndices<DIM + 1>(std::forward<OtherIdx>(others)...);
    }
    template <size_t DIM> FUNC_ATTR void printIndices() const {}

    template <size_t DIM, typename FirstIdx, typename... OtherIdx>
    FUNC_ATTR void printExtents(FirstIdx &&first, OtherIdx &&...others) const {
        printf(DIM > 0 ? ", %lld" : "%lld", (long long)(this->extent(DIM)));
        printExtents<DIM + 1>(std::forward<OtherIdx>(others)...);
    }
    template <size_t DIM> FUNC_ATTR void printExtents() const {}

  public:
    template <typename... Args>
    FUNC_ATTR constexpr mdspan_dbg(Args &&...args)
        : BaseClass(std::forward<Args>(args)...) {}

    template <typename... Args>
    FUNC_ATTR constexpr auto &&operator()(Args &&...args) {
        if (!checkDims<0>(args...)) {
            printf("Out of range access on index (");
            printIndices<0>(args...);
            printf("). The range is (");
            printExtents<0>(args...);
            printf(")\n");
            exit(-1);
        }
        return BaseClass::operator()(std::forward<Args>(args)...);
    }

    template <typename... Args>
    FUNC_ATTR constexpr auto &&operator()(Args &&...args) const {
        if (!checkDims<0>(args...)) {
            printf("Out of range access on index (");
            printIndices<0>(args...);
            printf("). The range is (");
            printExtents<0>(args...);
            printf(")\n");
            exit(-1);
        }
        return BaseClass::operator()(std::forward<Args>(args)...);
    }
};

// Use `restrict` pointers in mdspan
//
// https://github.com/kokkos/mdspan/issues/169

template <class ElementType> struct restrict_accessor {
    using offset_policy = stdex::default_accessor<ElementType>;
    using element_type = ElementType;
    using reference = ElementType &;
    using data_handle_type = ElementType *__restrict__;

    FUNC_ATTR constexpr restrict_accessor() noexcept = default;

    template <class OtherElementType>
#if (_MDSPAN_CPLUSPLUS >= MDSPAN_CXX_STD_20)
        requires(
            std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>)
#endif
    FUNC_ATTR constexpr restrict_accessor(
        restrict_accessor<OtherElementType>) noexcept {
    }

    FUNC_ATTR constexpr reference access(data_handle_type p,
                                         size_t i) const noexcept {
        return p[i];
    }
    FUNC_ATTR constexpr typename offset_policy::data_handle_type
    offset(data_handle_type p, size_t i) const noexcept {
        return p + i;
    }
};

template <class ElementType, class Extents>
using mdspan_r = stdex::mdspan<ElementType, Extents, stdex::layout_right,
                               restrict_accessor<ElementType>>;

// Convert mdspan to C array pointer
//
// The pointer type of array T[N][M][K] is (T*)[M][K], which is the type of a
// pointer to T[M][K], where M and K should be static extent
//
// This is required by OpenMP's Array Section grammar for its parallel reduction

template <class T, size_t... Ss> struct arr_t;
template <class T, size_t S, size_t... Ss> struct arr_t<T, S, Ss...> {
    static_assert(S != dynamic_extent,
                  "Dynamic extent is not supported for C array pointers");
    typedef typename arr_t<T, Ss...>::type type[S];
};
template <class T> struct arr_t<T> {
    typedef T type;
};

template <class T, size_t S, size_t... Ss> struct arr_ptr_t {
    typedef typename arr_t<T, Ss...>::type *type;
};

template <class T, class Layout, class Accessor, size_t... Ss>
auto toArrPtr(const mdspan<T, extents<Ss...>, Layout, Accessor> &s) {
    return (typename arr_ptr_t<T, Ss...>::type)s.data_handle();
}

#endif // FREE_TENSOR_MDSPAN_H
