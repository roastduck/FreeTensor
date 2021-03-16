#include <cstring>

#include <driver/array.h>
#include <driver/gpu.h>
#include <except.h>

namespace ir {

Array::Array(size_t nElem, DataType dtype, const Device &device)
    : size_(sizeOf(dtype) * nElem), nElem_(nElem), dtype_(dtype),
      device_(device) {
    switch (device_.type()) {
    case TargetType::CPU:
        ptr_ = new uint8_t[size_];
        break;
    case TargetType::GPU:
        checkCudaError(cudaMalloc(&ptr_, size_));
        break;
    default:
        ASSERT(false);
    }
}

Array::~Array() {
    if (ptr_ != nullptr) {
        switch (device_.type()) {
        case TargetType::CPU:
            delete[] ptr_;
            ptr_ = nullptr;
            break;
        case TargetType::GPU:
            cudaFree(ptr_);
            ptr_ = nullptr;
            break;
        }
    }
}

Array::Array(Array &&other)
    : ptr_(other.ptr_), size_(other.size_), nElem_(other.nElem_),
      dtype_(other.dtype_), device_(std::move(other.device_)) {
    other.ptr_ = nullptr; // MUST!
    other.size_ = 0;
}

Array &Array::operator=(Array &&other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    nElem_ = other.nElem_;
    dtype_ = other.dtype_;
    device_ = std::move(other.device_);
    other.ptr_ = nullptr; // MUST!
    other.size_ = 0;
    return *this;
}

void Array::fromCPU(const void *other, size_t size) {
    ASSERT(size == size_);
    ASSERT(ptr_ != nullptr);
    switch (device_.type()) {
    case TargetType::CPU:
        memcpy(ptr_, other, size_);
        break;
    case TargetType::GPU:
        checkCudaError(cudaMemcpy(ptr_, other, size_, cudaMemcpyDefault));
        break;
    default:
        ASSERT(false);
    }
}

void Array::toCPU(void *other, size_t size) {
    ASSERT(size == size_);
    ASSERT(ptr_ != nullptr);
    switch (device_.type()) {
    case TargetType::CPU:
        memcpy(other, ptr_, size_);
        break;
    case TargetType::GPU:
        checkCudaError(cudaMemcpy(other, ptr_, size_, cudaMemcpyDefault));
        break;
    default:
        ASSERT(false);
    }
}

} // namespace ir

