#ifndef FREE_TENSOR_CPU_RUNTIME_H
#define FREE_TENSOR_CPU_RUNTIME_H

#include <algorithm> // min, max
#include <array>     // ByValue
#include <atomic>
#include <cassert>
#include <cmath> // INFINITY, sqrt, exp
#include <cstdint>
#include <type_traits>

#include <omp.h>

#ifdef FT_WITH_MKL
#include <mkl.h>
#endif

#include "cpu_context.h"
#include "mdspan.h"
#include "unchecked_opt.h"

#define restrict __restrict__
#define __ByValArray std::array

inline auto floorDiv(std::integral auto a, std::integral auto b) {
    auto res = a / b;
    auto rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}

inline auto ceilDiv(std::integral auto a, std::integral auto b) {
    auto res = a / b;
    auto rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}

inline auto runtime_mod(std::integral auto a, std::integral auto b) {
    auto m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

template <class T> T runtime_square(T x) { return x * x; }

template <class T> T runtime_sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }

template <class T> void atomic_update(T &x, auto &&update) {
    // No need to keep the life time of `std::atomic_ref` outside this function:
    // Atomic operations applied to an object through an `std::atomic_ref` are
    // atomic with respect to atomic operations applied through any other
    // `std::atomic_ref` referencing the same object.
    std::atomic_ref<T> xAtomic(x);

    T xOld = xAtomic, y;
    do {
        y = update(xOld);
    } while (
        !xAtomic.compare_exchange_weak(xOld, y, std::memory_order_relaxed));
    // - `_weak` means we may fail even if `x` is unchanged, and we retry
    // - We can use a relaxed memory order: Since an `atomic_update` only
    // competes with other `atomic_update`s (FreeTensor's schedule ensures there
    // is no simultaneous `Load` and `ReduceTo` or simultaneous `Store` and
    // `ReduceTo`), and the only memory access in the loop of `atomic_update` is
    // this `compare_exchange`, we don't need to worry about the relative order
    // of this access with other accesses that cause side effect
}

#endif // FREE_TENSOR_CPU_RUNTIME_H
