#include <cstring>
#include <sstream>

#include <config.h>
#include <debug.h>
#include <driver/array.h>
#include <except.h>
#ifdef FT_WITH_CUDA
#include <driver/gpu.h>
#endif

namespace freetensor {

static std::string writeToBorrowedMsg(const Ref<Device> &lendingDev,
                                      const Ref<Device> &requestDev = nullptr) {
    std::ostringstream os;
    os << "Attempted to write ";
    if (requestDev.isValid()) {
        os << "from " << *requestDev << ", ";
    }
    os << "to a buffer borrowed from an external object at " << *lendingDev
       << ". Please either explicitly create a FreeTensor object by "
          "`ft.array`, which features automatic inter-device data transfer, or "
          "borrow from a object from the same device (e.g., from a "
          "correspoding PyTorch object)";
    return os.str();
}

static uint8_t *allocOn(size_t size, const Ref<Device> &device) {
    uint8_t *ptr = nullptr;
    switch (device->type()) {
    case TargetType::CPU:
        ptr = new uint8_t[size];
        break;

#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        if (!Config::debugCUDAWithUM()) {
            // Allocate traditional device memory. The performance is optimized
            // for well-scheduled computation, but debugging access from CPU to
            // GPU buffers is prohibitively slow
            checkCudaError(cudaSetDevice(device->num()));
            checkCudaError(cudaMalloc(&ptr, size));
        } else {
            // We allocate on CUDA's Unified Memory (UM), which features
            // automatically inter-device data transfer. Although we do explicit
            // memory copies here in `Array`, UM is helpful for debugging. As
            // long as explicitly copied, UM enjoys the same performance as
            // traditional device memory in kernel, but allocations and memory
            // copies may take longer time

            WARNING("Allocating on CUDA's Unified Memory. May negatively "
                    "affect allocation and memory copy performance");

            // Allocate a UM buffer
            checkCudaError(cudaMallocManaged(&ptr, size));

            // Set the buffer to prefer the selected device, so `cudaMemcpy`
            // with a `cudaMemcpyDefault` direction will treat it as a
            // traditional device memory
            checkCudaError(cudaMemAdvise(
                ptr, size, cudaMemAdviseSetPreferredLocation, device->num()));

            // `cudaMemset` is required, to actually move the buffer to the
            // device just set
            checkCudaError(cudaMemset(ptr, 0, size));
        }
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
            // Unlike `cudaMemcpy`, `cudaFree` does not synchronize the device
            // when the buffer to free was allocated by `cudaMallocAsync`
            // (https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ga042655cbbf3408f01061652a075e094).
            // The buffer to free may be being used by our generated code, at
            // any stream, or at host. If used by a stream, we need to
            // synchronize it before freeing. Our generated code make the
            // default stream (stream 0) synchronize with all other streams, so
            // synchronize with only stream 0 here is sufficient.
            cudaFreeAsync(ptr, 0);
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

static void copyOnSameDevice(void *dst, const void *src, size_t size,
                             const Ref<Device> &device) {
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

Array::Array(const std::vector<size_t> &shape, DataType dtype,
             bool dontDropBorrow, bool moved)
    : shape_(shape), dtype_(dtype), dontDropBorrow_(dontDropBorrow),
      moved_(moved) {
    nElem_ = 1;
    for (size_t dim : shape_) {
        nElem_ *= dim;
    }
    size_ = nElem_ * sizeOf(dtype_);
}

Array Array::moveFromRaw(void *ptr, const std::vector<size_t> &shape,
                         DataType dtype, const Ref<Device> &device) {
    Array ret(shape, dtype, false);
    ret.ptrs_ = {{device, (uint8_t *)ptr, false}};
    return ret;
}

Array Array::borrowFromRaw(void *ptr, const std::vector<size_t> &shape,
                           DataType dtype, const Ref<Device> &device,
                           bool dontDropBorrow, bool moved) {
    Array ret(shape, dtype, dontDropBorrow, moved);
    ret.ptrs_ = {{device, (uint8_t *)ptr, true}};
    return ret;
}

Array::~Array() {
    for (auto &&[device, ptr, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(ptr, device);
        }
    }
    if (tempPtr_.has_value()) {
        if (!tempPtr_->borrowed_) {
            freeFrom(tempPtr_->ptr_, tempPtr_->device_);
        } else {
            // Impossible, but we can't throw in a destructor
        }
    }
}

Array::Array(Array &&other)
    : ptrs_(std::move(other.ptrs_)), tempPtr_(std::move(other.tempPtr_)),
      size_(other.size_), nElem_(other.nElem_), shape_(std::move(other.shape_)),
      dtype_(other.dtype_), dontDropBorrow_(other.dontDropBorrow_),
      moved_(other.moved_) // No need to clear moved. Users are using Ref<Array>
                           // anyway
{
    other.ptrs_.clear(); // MUST!
    other.tempPtr_ = std::nullopt;
    other.size_ = 0;
}

Array &Array::operator=(Array &&other) {
    ptrs_ = std::move(other.ptrs_);
    tempPtr_ = std::move(other.tempPtr_);
    size_ = other.size_;
    nElem_ = other.nElem_;
    dtype_ = other.dtype_;
    dontDropBorrow_ = other.dontDropBorrow_;
    moved_ = other.moved_; // No need to clear moved_. Users are using
                           // Ref<Array> anyway
    other.ptrs_.clear();   // MUST!
    other.tempPtr_ = std::nullopt;
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
        ASSERT(false);
    } else {
        for (auto &&[d, p, _] : ptrs_) {
            if (d->type() == TargetType::CPU) {
                copyFromCPU(ptr, p, size_, device);
                goto done;
            }
        }
        copyFromCPU(ptr, rawSharedTo(Ref<Device>::make(TargetType::CPU)), size_,
                    device);
    }
done:
    ptrs_.emplace_back(device, ptr, false);
    return ptr;
}

void *Array::rawMovedTo(const Ref<Device> &device) {
    std::string err;
    for (auto [d, p, borrowed] : ptrs_) {
        if (*d == *device) {
            for (auto &&[_d, _p, _borrowed] : ptrs_) {
                if (_p != p && !_borrowed) {
                    freeFrom(_p, _d);
                } else if (dontDropBorrow_ && *d != *device) {
                    err = writeToBorrowedMsg(d, device);
                }
            }
            ptrs_ = {{d, p, borrowed}};
            if (!err.empty()) {
                throw InvalidIO(err);
            }
            return p;
        }
    }
    auto ptr = allocOn(size_, device);
    if (device->type() == TargetType::CPU) {
        for (auto &&[d, p, _] : ptrs_) {
            copyToCPU(ptr, p, size_, d);
            goto done;
        }
        ASSERT(false);
    } else {
        for (auto &&[d, p, _] : ptrs_) {
            if (d->type() == TargetType::CPU) {
                copyFromCPU(ptr, p, size_, device);
                goto done;
            }
        }
        copyFromCPU(ptr, rawSharedTo(Ref<Device>::make(TargetType::CPU)), size_,
                    device);
    }
done:
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(p, d);
        } else if (dontDropBorrow_ && *d != *device) {
            err = writeToBorrowedMsg(d, device);
        }
    }
    ptrs_ = {{device, ptr, false}};
    if (!err.empty()) {
        throw InvalidIO(err);
    }
    return ptr;
}

void *Array::rawInitTo(const Ref<Device> &device) {
    std::string err;
    for (auto [d, p, borrowed] : ptrs_) {
        if (*d == *device) {
            for (auto &&[_d, _p, _borrowed] : ptrs_) {
                if (_p != p && !_borrowed) {
                    freeFrom(_p, _d);
                } else if (dontDropBorrow_ && *d != *device) {
                    err = writeToBorrowedMsg(d, device);
                }
            }
            ptrs_ = {{d, p, borrowed}};
            if (!err.empty()) {
                throw InvalidIO(err);
            }
            return p;
        }
    }
    auto ptr = allocOn(size_, device);
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            freeFrom(p, d);
        } else if (dontDropBorrow_ && *d != *device) {
            err = writeToBorrowedMsg(d, device);
        }
    }
    ptrs_ = {{device, ptr, false}};
    if (!err.empty()) {
        throw InvalidIO(err);
    }
    return ptr;
}

void *Array::rawTemporarilyCopiedTo(const Ref<Device> &device) {
    if (tempPtr_.has_value()) {
        freeFrom(tempPtr_->ptr_, tempPtr_->device_);
        tempPtr_ = std::nullopt;
    }
    if (!tempPtr_.has_value()) {
        tempPtr_ = {{device, allocOn(size_, device), false}};
    }
    // We first call `rawSharedTo` to make a internal copy on the detination
    // target. So if we are repeatly calling `rawTemporarilyCopiedTo` to another
    // device, we don't always do cross-device copying
    copyOnSameDevice(tempPtr_->ptr_, rawSharedTo(device), size_, device);
    return tempPtr_->ptr_;
}

void Array::makePrivateCopy() {
    std::string err;
    std::vector<ArrayCopy> newPtrs;
    newPtrs.reserve(ptrs_.size());
    for (auto &&[d, p, borrowed] : ptrs_) {
        if (!borrowed) {
            newPtrs.emplace_back(d, p, borrowed);
        } else if (dontDropBorrow_) {
            err = writeToBorrowedMsg(d);
        }
    }
    if (!newPtrs.empty()) {
        ptrs_ = std::move(newPtrs);
    } else {
        for (auto &&[d, p, _] : ptrs_) {
            auto dev = Ref<Device>::make(TargetType::CPU);
            auto ptr = allocOn(size_, dev);
            copyToCPU(ptr, p, size_, d);
            ptrs_ = {{dev, ptr, false}};
        }
    }
    if (!err.empty()) {
        throw InvalidIO(err);
    }
}

} // namespace freetensor
