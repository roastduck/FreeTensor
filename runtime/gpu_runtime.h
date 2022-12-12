#ifndef FREE_TENSOR_GPU_RUNTIME_H
#define FREE_TENSOR_GPU_RUNTIME_H

#include <algorithm>
#include <assert.h>
#include <cmath> // INFINITY
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "gpu_context.h"

#include "integer_range.h"
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
        checkCudaError(cudaMalloc(&ptr, size));
    }
    return ptr;
}

template <class T, size_t n> struct __ByValArray {
    T data[n];
    __host__ __device__ const T &operator[](size_t i) const { return data[i]; }
    __host__ __device__ T &operator[](size_t i) { return data[i]; }
};

// Access a scalar on GPU, only for debugging
template <class T> class GPUScalar {
    T *ptr_;

  public:
    explicit GPUScalar(T *ptr) : ptr_(ptr) {}
    explicit GPUScalar(T &ref) : ptr_(&ref) {}

    // Our compiler passes ensures the correctness, and the performance is not a
    // problem because it is for debugging
    explicit GPUScalar(const T *ptr) : ptr_(const_cast<T *>(ptr)) {}
    explicit GPUScalar(const T &ref) : ptr_(const_cast<T *>(&ref)) {}

    operator T() const {
        T ret;
        checkCudaError(cudaMemcpy(&ret, ptr_, sizeof(T), cudaMemcpyDefault));
        return ret;
    }

    GPUScalar &operator=(const T &other) {
        checkCudaError(cudaMemcpy(ptr_, &other, sizeof(T), cudaMemcpyDefault));
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

// NVCC does not support C++20
template <class T, typename std::enable_if_t<std::is_integral_v<T>> * = nullptr>
__host__ __device__ T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}
template <class T, typename std::enable_if_t<std::is_integral_v<T>> * = nullptr>
__host__ __device__ T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
}
template <class T, typename std::enable_if_t<std::is_integral_v<T>> * = nullptr>
__host__ __device__ T runtime_mod(T a, T b) {
    T m = a % b;
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

template <class T> __host__ __device__ T runtime_square(T x) { return x * x; }

template <class T> __host__ __device__ T runtime_sigmoid(T x) {
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
