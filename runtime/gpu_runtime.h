#ifndef FREE_TENSOR_GPU_RUNTIME_H
#define FREE_TENSOR_GPU_RUNTIME_H

#include <algorithm>
#include <assert.h>
#include <cmath> // INFINITY
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "gpu_context.h"

#include "mdspan.h"
#include "unchecked_opt.h"

#include "../3rd-party/cuda-samples/Common/helper_math.h"

#define restrict __restrict__

#define checkCudaError(call)                                                   \
    {                                                                          \
        auto err = (call);                                                     \
        if (cudaSuccess != err) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            throw std::runtime_error("cuda error");                            \
        }                                                                      \
    }

inline void *cudaNew(size_t size) {
    void *ptr = nullptr;
    if (size > 0) {
#ifndef FT_DEBUG_CUDA_WITH_UM
        checkCudaError(cudaMalloc(&ptr, size));
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

#endif // FREE_TENSOR_GPU_RUNTIME_H
