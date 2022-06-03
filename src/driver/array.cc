#include <cstring>

#include <config.h>
#include <debug.h>
#include <driver/array.h>
#include <except.h>
#ifdef FT_WITH_CUDA
#include <driver/gpu.h>
#endif

namespace freetensor {

static uint8_t *allocOn(size_t size, const Ref<Device> &device) {
    uint8_t *ptr = nullptr;
    switch (device->type()) {
    case TargetType::CPU:
        ptr = new uint8_t[size];
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        checkCudaError(cudaMalloc(&ptr, size));
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
    return ptr;
}

static void freeFrom(uint8_t *&ptr, const Ref<Device> &device) {
    if (ptr != nullptr) {
        switch (device->type()) {
        case TargetType::CPU:
            delete[] ptr;
            ptr = nullptr;
            break;
#ifdef FT_WITH_CUDA
        case TargetType::GPU:
            cudaFree(ptr);
            ptr = nullptr;
            break;
#endif // FT_WITH_CUDA
        default:;
            // Do nothing. We can't throw error in a destructor
        }
        ptr = nullptr;
    }
}

static void copyFromCPU(void *dst /* Any device */, const void *src /* CPU */,
                        size_t size, const Ref<Device> &device) {
    ASSERT(dst != nullptr);
    ASSERT(src != nullptr);
    switch (device->type()) {
    case TargetType::CPU:
        memcpy(dst, src, size);
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

static void copyToCPU(void *dst /* CPU */, const void *src /* Any device */,
                      size_t size, const Ref<Device> &device) {
    ASSERT(dst != nullptr);
    ASSERT(src != nullptr);
    switch (device->type()) {
    case TargetType::CPU:
        memcpy(dst, src, size);
        break;
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDefault));
        break;
#endif // FT_WITH_CUDA
    default:
        ASSERT(false);
    }
}

Array::Array(const std::vector<size_t> &shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    nElem_ = 1;
    for (size_t dim : shape_) {
        nElem_ *= dim;
    }
    size_ = nElem_ * sizeOf(dtype_);
}

Array Array::moveFromRaw(void *ptr, const std::vector<size_t> &shape,
                         DataType dtype, const Ref<Device> &device) {
    Array ret(shape, dtype);
    ret.ptrs_ = {{device, (uint8_t *)ptr, false}};
    return ret;
}

Array Array::borrowFromRaw(void *ptr, const std::vector<size_t> &shape,
                           DataType dtype, const Ref<Device> &device) {
    Array ret(shape, dtype);
    ret.ptrs_ = {{device, (uint8_t *)ptr, true}};
    return ret;
}

Array::~Array() {
    for (auto &&[device, ptr, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(ptr, device);
        }
    }
}

Array::Array(Array &&other)
    : ptrs_(std::move(other.ptrs_)), size_(other.size_), nElem_(other.nElem_),
      shape_(std::move(other.shape_)), dtype_(other.dtype_) {
    other.ptrs_.clear(); // MUST!
    other.size_ = 0;
}

Array &Array::operator=(Array &&other) {
    ptrs_ = std::move(other.ptrs_);
    size_ = other.size_;
    nElem_ = other.nElem_;
    dtype_ = other.dtype_;
    other.ptrs_.clear(); // MUST!
    other.size_ = 0;
    return *this;
}

void *Array::rawSharedTo(const Ref<Device> &device) {
    for (auto &&[d, p, _] : ptrs_) {
        if (*d == *device) {
            return p;
        }
    }
    auto ptr = allocOn(size_, device);
    if (device->type() == TargetType::CPU) {
        for (auto &&[d, p, _] : ptrs_) {
            copyToCPU(ptr, p, size_, d);
            goto done;
        }
    } else {
        for (auto &&[d, p, _] : ptrs_) {
            if (d->type() == TargetType::CPU) {
                copyFromCPU(ptr, p, size_, device);
                goto done;
            }
        }
        copyFromCPU(ptr, rawSharedTo(Ref<Device>::make(Ref<CPU>::make())),
                    size_, device);
    }
done:
    ptrs_.emplace_back(device, ptr, false);
    return ptr;
}

void *Array::rawMovedTo(const Ref<Device> &device) {
    for (auto [d, p, borrowed] : ptrs_) {
        if (*d == *device) {
            for (auto &&[_d, _p, _borrowed] : ptrs_) {
                if (_p != p && !_borrowed) {
                    freeFrom(_p, _d);
                }
            }
            ptrs_ = {{d, p, borrowed}};
            return p;
        }
    }
    auto ptr = allocOn(size_, device);
    if (device->type() == TargetType::CPU) {
        for (auto &&[d, p, _] : ptrs_) {
            copyToCPU(ptr, p, size_, d);
            goto done;
        }
    } else {
        for (auto &&[d, p, _] : ptrs_) {
            if (d->type() == TargetType::CPU) {
                copyFromCPU(ptr, p, size_, device);
                goto done;
            }
        }
        copyFromCPU(ptr, rawSharedTo(Ref<Device>::make(Ref<CPU>::make())),
                    size_, device);
    }
done:
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(p, d);
        }
    }
    ptrs_ = {{device, ptr, false}};
    return ptr;
}

void *Array::rawInitTo(const Ref<Device> &device) {
    for (auto [d, p, borrowed] : ptrs_) {
        if (*d == *device) {
            for (auto &&[_d, _p, _borrowed] : ptrs_) {
                if (_p != p && !_borrowed) {
                    freeFrom(_p, _d);
                }
            }
            ptrs_ = {{d, p, borrowed}};
            return p;
        }
    }
    auto ptr = allocOn(size_, device);
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(p, d);
        }
    }
    ptrs_ = {{device, ptr, false}};
    return ptr;
}

void Array::makePrivateCopy() {
    std::vector<ArrayCopy> newPtrs;
    newPtrs.reserve(ptrs_.size());
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            newPtrs.emplace_back(d, p, borrowed);
        }
    }
    if (!newPtrs.empty()) {
        ptrs_ = std::move(newPtrs);
        return;
    }

    for (auto &&[d, p, _] : ptrs_) {
        auto dev = Ref<Device>::make(Ref<CPU>::make());
        auto ptr = allocOn(size_, dev);
        copyToCPU(ptr, p, size_, d);
        ptrs_ = {{dev, ptr, false}};
        return;
    }
}

} // namespace freetensor
