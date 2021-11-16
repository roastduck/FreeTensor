#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <algorithm>
#include <assert.h>
#include <cmath> // INFINITY
#include <cstdint>

#include "gpu_context.h"

#include "../3rd-party/cuda-samples/Common/helper_math.h"

#define restrict __restrict__

template <class T, size_t n> struct __ByValArray {
    T data[n];
    __host__ __device__ const T &operator[](size_t i) const { return data[i]; }
    __host__ __device__ T &operator[](size_t i) { return data[i]; }
};

template <class T> __host__ __device__ T floorDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res - (rem != 0 && ((rem < 0) != (b < 0)));
}
template <class T> __host__ __device__ T ceilDiv(T a, T b) {
    T res = a / b, rem = a % b;
    return res + (rem != 0 && ((rem < 0) == (b < 0)));
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

#endif // GPU_RUNTIME_H
