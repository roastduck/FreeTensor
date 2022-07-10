#ifndef FREE_TENSOR_OMP_UTILS_H
#define FREE_TENSOR_OMP_UTILS_H

#include <atomic>
#include <concepts>
#include <exception>
#include <functional>

#include <omp.h>

namespace freetensor {

/**
 * Wrapper for thread safe OpenMP parallel for
 */
template <std::integral T>
void exceptSafeParallelFor(T begin, T end, T step,
                           const std::function<void(T)> &body,
                           omp_sched_t schedKind,
                           int schedChunkSize = 0 /* 0 = auto */) {
    std::atomic_flag hasExcept_ = ATOMIC_FLAG_INIT;
    // ATOMIC_FLAG_INIT is required before C++20, but deprecated since C++20.
    // Keep it for now for potentially unsupported compilers

    std::exception_ptr except_;

    omp_set_schedule(schedKind, schedChunkSize);
    // omp cancel requires separating `omp parallel` and `omp for`:
    // https://stackoverflow.com/questions/30275513/gcc-5-1-warns-cancel-construct-within-parallel-for-construct
#pragma omp parallel
    {
        // schedule(runtime) = read schedule set by omp_set_schedule
#pragma omp for schedule(runtime)
        for (auto i = begin; i < end; i += step) {
            try {
                body(i);
            } catch (...) {
                if (!hasExcept_.test_and_set()) {
                    except_ = std::current_exception();
#pragma omp cancel for
                }
            }
#pragma omp cancellation point for // Check wheter to cancel here
        }
    }
    if (except_) {
        std::rethrow_exception(except_);
    }
}

} // namespace freetensor

#endif // FREE_TENSOR_OMP_UTILS_H
