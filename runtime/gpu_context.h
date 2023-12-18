#ifndef FREE_TENSOR_GPU_CONTEXT_H
#define FREE_TENSOR_GPU_CONTEXT_H

#include <cublas_v2.h>
#include <iostream>

#include "context.h"

#define runtimeCheckCudaError(call)                                            \
    {                                                                          \
        auto err = (call);                                                     \
        if (cudaSuccess != err) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            throw std::runtime_error("cuda error");                            \
        }                                                                      \
    }

#define checkCublasError(call)                                                 \
    {                                                                          \
        auto err = (call);                                                     \
        if (CUBLAS_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "cuBLAS error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cublasGetErrorString(err));            \
            throw std::runtime_error("cublas error");                          \
        }                                                                      \
    }

inline const char *cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

// checkCutlassError is defined in gpu_runtime.h

class GPUContext : public Context {
    bool initialized_ = false;
    cublasHandle_t cublas_;
    void *gpuGlobalStaticPool_ = nullptr;
    cudaMemPool_t gpuGlobalDynamicPool_ = nullptr;

  public:
    GPUContext(int deviceId, uint64_t gpuGlobalStaticPoolSize, bool useUM) {
        // Wait for any unfinished cudaFreeAsync. This seems unnecessary
        // according to CUDA's document, but our tests need it. (TODO:
        // Remove it)
        runtimeCheckCudaError(cudaDeviceSynchronize());

        runtimeCheckCudaError(cudaSetDevice(deviceId));

        checkCublasError(cublasCreate(&cublas_));

        if (gpuGlobalStaticPoolSize > 0) {
            if (!useUM) {
                runtimeCheckCudaError(
                    cudaMalloc(&gpuGlobalStaticPool_, gpuGlobalStaticPoolSize));
            } else {
                // Please refer to src/driver/array.cc:allocOn for details
                runtimeCheckCudaError(cudaMallocManaged(
                    &gpuGlobalStaticPool_, gpuGlobalStaticPoolSize));
                runtimeCheckCudaError(
                    cudaMemAdvise(gpuGlobalStaticPool_, gpuGlobalStaticPoolSize,
                                  cudaMemAdviseSetPreferredLocation, deviceId));
                runtimeCheckCudaError(cudaMemset(gpuGlobalStaticPool_, 0,
                                                 gpuGlobalStaticPoolSize));
            }
        }

        cudaMemPoolProps mem_pool_props = {};
        mem_pool_props.allocType = cudaMemAllocationTypePinned;
        mem_pool_props.location.type = cudaMemLocationTypeDevice;
        mem_pool_props.location.id = deviceId;
        runtimeCheckCudaError(
            cudaMemPoolCreate(&gpuGlobalDynamicPool_, &mem_pool_props));
        uint64_t threshold = UINT64_MAX; // Never actively free to OS, but other
                                         // process may steal memory
        runtimeCheckCudaError(cudaMemPoolSetAttribute(
            gpuGlobalDynamicPool_, cudaMemPoolAttrReleaseThreshold,
            &threshold));

        initialized_ = true;
    }
    ~GPUContext() {
        if (initialized_) {
            cublasDestroy(cublas_);
            if (gpuGlobalStaticPool_ != nullptr) {
                cudaFree(gpuGlobalStaticPool_);
            }
            cudaMemPoolDestroy(gpuGlobalDynamicPool_);
            initialized_ = false;
        }
    }

    GPUContext(const GPUContext &) = delete;
    GPUContext &operator=(const GPUContext &) = delete;

    GPUContext(GPUContext &&other)
        : initialized_(other.initialized_), cublas_(other.cublas_),
          gpuGlobalStaticPool_(other.gpuGlobalStaticPool_),
          gpuGlobalDynamicPool_(other.gpuGlobalDynamicPool_) {
        other.initialized_ = false;
    }
    GPUContext &operator=(GPUContext &&other) {
        initialized_ = other.initialized_;
        cublas_ = other.cublas_;
        gpuGlobalStaticPool_ = other.gpuGlobalStaticPool_;
        gpuGlobalDynamicPool_ = other.gpuGlobalDynamicPool_;
        other.initialized_ = false;
        return *this;
    }

    cublasHandle_t cublas() const { return cublas_; }
    void *gpuGlobalStaticPool() const { return gpuGlobalStaticPool_; }
    cudaMemPool_t gpuGlobalDynamicPool() const { return gpuGlobalDynamicPool_; }
};

extern "C" typedef GPUContext *GPUContext_t;

#endif // FREE_TENSOR_GPU_CONTEXT_H
