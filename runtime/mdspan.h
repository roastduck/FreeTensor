#ifndef FREE_TENSOR_MDSPAN_H
#define FREE_TENSOR_MDSPAN_H

#include <cstdint>

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
    requires(std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>)
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
template <class T> struct arr_t<T> { typedef T type; };

template <class T, size_t S, size_t... Ss> struct arr_ptr_t {
    typedef typename arr_t<T, Ss...>::type *type;
};

template <class T, class Layout, class Accessor, size_t... Ss>
auto toArrPtr(const mdspan<T, extents<Ss...>, Layout, Accessor> &s) {
    return (typename arr_ptr_t<T, Ss...>::type)s.data_handle();
}

#endif // FREE_TENSOR_MDSPAN_H
