#ifndef FREE_TENSOR_GPU_RUNTIME_H
#define FREE_TENSOR_GPU_RUNTIME_H

#include <algorithm>
#include <assert.h>
#include <cmath> // INFINITY
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <type_traits>

#include "gpu_context.h"

#include "mdspan.h"
#include "unchecked_opt.h"

#include "../3rd-party/cuda-samples/Common/helper_math.h"

#define restrict __restrict__

#define checkCudaError(...) runtimeCheckCudaError(__VA_ARGS__)

inline void *cudaNew(size_t size, cudaStream_t stream) {
    void *ptr = nullptr;
    if (size > 0) {
#ifndef FT_DEBUG_CUDA_WITH_UM
        checkCudaError(cudaMallocAsync(&ptr, size, stream));
#else
        // Please refer to src/driver/array.cc:allocOn for details
        int device;
        checkCudaError(cudaGetDevice(&device));
        checkCudaError(cudaMallocManaged(&ptr, size));
        checkCudaError(cudaMemAdvise(
            ptr, size, cudaMemAdviseSetPreferredLocation, device));
        checkCudaError(cudaMemset(ptr, 0, size));
#endif // FT_DEBUG_CUDA_WITH_UM
    }
    return ptr;
}

inline void *cudaNewFromPool(size_t size, cudaStream_t stream,
                             cudaMemPool_t pool) {
    void *ptr = nullptr;
    if (size > 0) {
#ifndef FT_DEBUG_CUDA_WITH_UM
        checkCudaError(cudaMallocFromPoolAsync(&ptr, size, pool, stream));
#else
        // Fallback
        return cudaNew(size, stream);
#endif // FT_DEBUG_WITH_UM
    }
    return ptr;
}

template <class T, size_t n> struct __ByValArray {
    T data[n];
    __host__ __device__ const T &operator[](size_t i) const { return data[i]; }
    __host__ __device__ T &operator[](size_t i) { return data[i]; }
};

/**
 * Base class for `GPUScalar`. Used for type traits only
 */
class GPUScalarBase {};

/**
 * Access a scalar on GPU, only for debugging
 *
 * For any scalar type `T`, `GPUScalar<T>` represents such a scalar store in GPU
 * memory. Reading from a `GPUScalar<T>` or performing any arithmetic operation
 * on a `GPUScalar<T>` invokes a single-scalar `cudaMemcpy` and returns a `T`.
 * Writing into a `GPUScalar<T>` also invokes a single-scalar `cudaMemcpy`.
 *
 * NOTE: Since arithmetic operations on `GPUScalar<T>` returns `T`, generic math
 * functions like `abs` defined below should return `auto` (instead of returning
 * the same type with its arguments)
 */
template <class T> class GPUScalar : public GPUScalarBase {
    T *ptr_;

  private:
    bool isUnifiedMemory() const {
        // Check memory location at run time. Even if we have
        // FT_DEBUG_CUDA_USE_UM, the pointer is not guaranteed to be pointing at
        // UM. It can be a memory view from external inputs
        cudaPointerAttributes attr;
        checkCudaError(cudaPointerGetAttributes(&attr, ptr_));
        switch (attr.type) {
        case cudaMemoryTypeDevice:
            return false;
        case cudaMemoryTypeManaged:
            return true;
        default:
            fprintf(stderr, "Unexpcted memory location\n");
            exit(-1);
        }
    }

  public:
    typedef T ScalarType;

    explicit GPUScalar(T *ptr) : ptr_(ptr) {}
    explicit GPUScalar(T &ref) : ptr_(&ref) {}

    // Our compiler passes ensures the correctness, and the performance is not a
    // problem because it is for debugging
    explicit GPUScalar(const T *ptr) : ptr_(const_cast<T *>(ptr)) {}
    explicit GPUScalar(const T &ref) : ptr_(const_cast<T *>(&ref)) {}

    operator T() const {
        if (isUnifiedMemory()) {
            return *ptr_;
        } else {
            T ret;
            checkCudaError(
                cudaMemcpy(&ret, ptr_, sizeof(T), cudaMemcpyDefault));
            return ret;
        }
    }

    GPUScalar &operator=(const GPUScalar &other) { return *this = (T)other; }
    GPUScalar &operator=(GPUScalar &&other) { return *this = (T)other; }
    GPUScalar &operator=(const T &other) {
        if (isUnifiedMemory()) {
            *ptr_ = other;
        } else {
            checkCudaError(
                cudaMemcpy(ptr_, &other, sizeof(T), cudaMemcpyDefault));
        }
        return *this;
    }

    friend T operator+(const GPUScalar &lhs, const GPUScalar &rhs) {
        return T(lhs) + T(rhs);
    }
    friend T operator-(const GPUScalar &lhs, const GPUScalar &rhs) {
        return T(lhs) - T(rhs);
    }
    friend T operator*(const GPUScalar &lhs, const GPUScalar &rhs) {
        return T(lhs) * T(rhs);
    }
    friend T operator/(const GPUScalar &lhs, const GPUScalar &rhs) {
        return T(lhs) / T(rhs);
    }

    GPUScalar &operator+=(const T &other) { return *this = *this + other; }
    GPUScalar &operator-=(const T &other) { return *this = *this - other; }
    GPUScalar &operator*=(const T &other) { return *this = *this * other; }
    GPUScalar &operator/=(const T &other) { return *this = *this / other; }
};
template <class T> GPUScalar<T> gpuScalar(T *ptr) { return GPUScalar<T>(ptr); }
template <class T> GPUScalar<T> gpuScalar(T &ref) { return GPUScalar<T>(ref); }
template <class T> GPUScalar<T> gpuScalar(const T *ptr) {
    return GPUScalar<T>(ptr);
}
template <class T> GPUScalar<T> gpuScalar(const T &ref) {
    return GPUScalar<T>(ref);
}

// TODO: Update to C++20 concepts after CUDA 12 lands

// TODO: When C++20 is ready, replace the following with:
//
// template <typename T>
// concept IsIntegralAnywhere = std::is_integral_v<T> ||
//     (std::is_base_of_v<GPUScalarBase, T>
//          &&std::is_integral_v<typename T::ScalarType>);
template <typename T, typename = void>
constexpr bool IsIntegralAnywhere = false;
template <typename T>
constexpr bool IsIntegralAnywhere<T, std::enable_if_t<std::is_integral_v<T>>> =
    true;
template <typename T>
constexpr bool IsIntegralAnywhere<
    T, std::enable_if_t<std::is_base_of_v<GPUScalarBase, T> &&
                        std::is_integral_v<typename T::ScalarType>>> = true;

template <class T, class U,
          typename std::enable_if_t<IsIntegralAnywhere<T> &&
                                    IsIntegralAnywhere<U>> * = nullptr>
__host__ __device__ auto floorDiv(T a, U b) {
    auto res = a / b;
    auto rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}

template <class T, class U,
          typename std::enable_if_t<IsIntegralAnywhere<T> &&
                                    IsIntegralAnywhere<U>> * = nullptr>
__host__ __device__ auto ceilDiv(T a, U b) {
    auto res = a / b;
    auto rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}

// Below are overloading of math functions. Although CUDA claims "many"
// functions are already overloaded in
// https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH.html, but it
// seems they never gave a full list of functions that we can safely use
// (without implicitly converting to double). So we overload them explicitly
// here

template <class T, class U,
          typename std::enable_if_t<IsIntegralAnywhere<T> &&
                                    IsIntegralAnywhere<U>> * = nullptr>
__host__ __device__ auto runtime_mod(T a, U b) {
    auto m = a % b;
    if (m < 0) {
        // m += (b < 0) ? -b : b; // avoid this form: it is UB when b == INT_MIN
        m = (b < 0) ? m - b : m + b;
    }
    return m;
}

inline __host__ __device__ int4 make_int4(int4 a) { return a; }
inline __host__ __device__ int2 make_int2(int2 a) { return a; }
inline __host__ __device__ float4 make_float4(float4 a) { return a; }
inline __host__ __device__ float2 make_float2(float2 a) { return a; }

inline __host__ __device__ float runtime_sqrt(float x) { return sqrtf(x); }
inline __host__ __device__ double runtime_sqrt(double x) { return sqrt(x); }

inline __host__ __device__ float runtime_exp(float x) { return expf(x); }
inline __host__ __device__ double runtime_exp(double x) { return exp(x); }

inline __host__ __device__ float runtime_log(float x) { return logf(x); }
inline __host__ __device__ double runtime_log(double x) { return log(x); }

inline __host__ __device__ float runtime_sin(float x) { return sinf(x); }
inline __host__ __device__ double runtime_sin(double x) { return sin(x); }

inline __host__ __device__ float runtime_cos(float x) { return cosf(x); }
inline __host__ __device__ double runtime_cos(double x) { return cos(x); }

inline __host__ __device__ float runtime_tan(float x) { return tanf(x); }
inline __host__ __device__ double runtime_tan(double x) { return tan(x); }

inline __host__ __device__ float runtime_tanh(float x) { return tanhf(x); }
inline __host__ __device__ double runtime_tanh(double x) { return tanh(x); }

template <class T> __host__ __device__ auto runtime_square(T x) {
    return x * x;
}

template <class T> __host__ __device__ auto runtime_sigmoid(T x) {
    return 1.0 / (1.0 + runtime_exp(-x));
}

inline __host__ __device__ float runtime_abs(float x) { return fabsf(x); }
inline __host__ __device__ double runtime_abs(double x) { return fabs(x); }

inline __host__ __device__ float runtime_floor(float x) { return floorf(x); }
inline __host__ __device__ double runtime_floor(double x) { return floor(x); }

inline __host__ __device__ float runtime_ceil(float x) { return ceilf(x); }
inline __host__ __device__ double runtime_ceil(double x) { return ceil(x); }

inline __host__ __device__ int clz(unsigned int x) {
#if defined(__CUDA_ARCH__)
    return __clz((int)x);
#else
    return __builtin_clz(x);
#endif
}
inline __host__ __device__ int clzll(unsigned long long x) {
#if defined(__CUDA_ARCH__)
    return __clzll((long long)x);
#else
    return __builtin_clz(x);
#endif
}

// TODO: Switch to concept after C++20 in CUDA 12
template <typename T, typename = void> struct SameSizeUnsigned;
template <typename T>
struct SameSizeUnsigned<T, std::enable_if_t<sizeof(T) == 2>> {
    typedef unsigned short type;
};
template <typename T>
struct SameSizeUnsigned<T, std::enable_if_t<sizeof(T) == 4>> {
    typedef unsigned type;
};
template <typename T>
struct SameSizeUnsigned<T, std::enable_if_t<sizeof(T) == 8>> {
    typedef unsigned long long type;
};
template <typename T>
using SameSizeUnsignedT = typename SameSizeUnsigned<T>::type;

template <typename T, typename Func, // Can't use std::function because it lost
                                     // __device__ annotation
          typename std::enable_if_t<sizeof(T) == 2 || sizeof(T) == 4 ||
                                    sizeof(T) == 8> * = nullptr>
__host__ __device__ void atomicUpdate(T &x, Func &&update) {
    typedef SameSizeUnsignedT<T> U;
    T xOld = x;
    U xOldBits = *((U *)&xOld);
    // TODO: Use `bit_cast` after we have C++20 in CUDA 12
    while (true) {
        T y = update(xOld);
        U yBits = *((U *)&y);
        // TODO: Use `bit_cast` after we have C++20 in CUDA 12
        U xNewBits = atomicCAS((U *)&x, xOldBits, yBits);
        if (xNewBits == xOldBits) {
            break;
        } else {
            xOldBits = xNewBits;
            xOld = *((T *)&xNewBits);
            // TODO: Use `bit_cast` after we have C++20 in CUDA 12
        }
    }
    // We are OK with CUDA's relaxed memory order. Since an `atomicUpdate` only
    // competes with other `atomicUpdate`s (FreeTensor's schedule ensures there
    // is no simultaneous `Load` and `ReduceTo` or simultaneous `Store` and
    // `ReduceTo`), and the only memory access in the loop of `atomicUpdate` is
    // this `atomicCAS`, we don't need to worry about the relative order of this
    // access with other accesses that cause side effect
}

template <typename T, typename = void>
__host__ __device__ void runtimeAtomicMin(T *addr, T val) {
    atomicUpdate(*addr, [&](T x) { return min(x, val); });
}
template <typename T, std::enable_if_t<std::is_integral_v<T>>>
__host__ __device__ void runtimeAtomicMin(T *addr, T val) {
    atomicMin(addr, val); // Only defined for integers
}

template <typename T, typename = void>
__host__ __device__ void runtimeAtomicMax(T *addr, T val) {
    atomicUpdate(*addr, [&](T x) { return max(x, val); });
}
template <typename T, std::enable_if_t<std::is_integral_v<T>>>
__host__ __device__ void runtimeAtomicMax(T *addr, T val) {
    atomicMax(addr, val); // Only defined for integers
}

#endif // FREE_TENSOR_GPU_RUNTIME_H
