#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <algorithm>
#include <assert.h>
#include <cmath> // INFINITY
#include <cstdint>

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

#endif // GPU_RUNTIME_H
