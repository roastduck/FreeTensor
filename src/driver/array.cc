#include <cstring>

#include <driver/array.h>
#include <driver/gpu.h>
#include <except.h>

namespace ir {

Array::Array(const std::vector<size_t> &shape, DataType dtype,
             const std::string &device)
    : dtype_(dtype), shape_(shape), device_(device) {
    size_ = sizeOf(dtype_);
    for (size_t dim : shape_) {
        size_ *= dim;
    }

    if (device_ == "cpu") {
        ptr_ = new uint8_t[size_];
    } else if (device_ == "gpu") {
        checkCudaError(cudaMalloc(&ptr_, size_));
    } else {
        throw DriverError("Unsupported device " + device_);
    }
}

Array::~Array() {
    if (ptr_ != nullptr) {
        if (device_ == "cpu") {
            delete[] ptr_;
            ptr_ = nullptr;
        } else if (device_ == "gpu") {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
}

void Array::fromCPU(const void *other, size_t size) {
    if (size != size_) {
        throw DriverError("Size of the target buffer not the same");
    }
    if (device_ == "cpu") {
        memcpy(ptr_, other, size_);
    } else if (device_ == "gpu") {
        checkCudaError(cudaMemcpy(ptr_, other, size_, cudaMemcpyDefault));
    } else {
        throw DriverError("Unsupported device " + device_);
    }
}

void Array::toCPU(void *other, size_t size) {
    if (size != size_) {
        throw DriverError("Size of the target buffer not the same");
    }
    if (device_ == "cpu") {
        memcpy(other, ptr_, size_);
    } else if (device_ == "gpu") {
        checkCudaError(cudaMemcpy(other, ptr_, size_, cudaMemcpyDefault));
    } else {
        throw DriverError("Unsupported device " + device_);
    }
}

} // namespace ir

