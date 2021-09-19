#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include <cublas_v2.h>
#include <iostream>

#define checkCublasError(call)                                                 \
    {                                                                          \
        auto err = call;                                                       \
        if (CUBLAS_STATUS_SUCCESS != err) {                                    \
            fprintf(stderr, "cuBLAS error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cublasGetErrorString(err));            \
            exit(EXIT_FAILURE);                                                \
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

class GPUContext {
    cublasHandle_t cublas_;

  public:
    GPUContext() { checkCublasError(cublasCreate(&cublas_)); }
    ~GPUContext() { checkCublasError(cublasDestroy(cublas_)); }

    cublasHandle_t cublas() const { return cublas_; }
};

extern "C" typedef GPUContext *GPUContext_t;

#endif // GPU_CONTEXT_H
