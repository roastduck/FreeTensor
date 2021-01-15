#ifndef GPU_H
#define GPU_H

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

#endif // GPU_H
