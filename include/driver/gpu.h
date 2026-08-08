#ifndef FREE_TENSOR_GPU_H
#define FREE_TENSOR_GPU_H

#include <sstream>

#ifdef FT_WITH_CUDA
#include <cuda_runtime.h>

#include <except.h>

#define checkCudaError(call)                                                   \
    {                                                                          \
        auto err = (call);                                                     \
        if (cudaSuccess != err) {                                              \
            throw DriverError(cudaGetErrorString(err));                        \
        }                                                                      \
    }

#if CUDART_VERSION >= 13000
namespace freetensor {
inline cudaMemLocation cudaMemAdviseLocationForDevice(int device) {
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    return location;
}
} // namespace freetensor
#else
namespace freetensor {
inline int cudaMemAdviseLocationForDevice(int device) { return device; }
} // namespace freetensor
#endif

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_H
