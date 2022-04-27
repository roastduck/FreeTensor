#ifndef FREE_TENSOR_GPU_H
#define FREE_TENSOR_GPU_H

#include <sstream>

#include <cuda_runtime.h>

#include <except.h>

#define checkCudaError(call)                                                   \
    {                                                                          \
        auto err = (call);                                                     \
        if (cudaSuccess != err) {                                              \
            throw DriverError(cudaGetErrorString(err));                        \
        }                                                                      \
    }

#endif // FREE_TENSOR_GPU_H
