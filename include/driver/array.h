#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>

#include <driver/device.h>
#include <tensor.h>

namespace ir {

class Array {
    uint8_t *ptr_ = nullptr;
    size_t size_ = 0, nElem_ = 0;
    DataType dtype_;
    Device device_;

  public:
    Array(size_t nElem, DataType dtype, const Device &device);

    // Move from raw pointer. Use with cautious
    Array(void *ptr, size_t size, DataType dtype, const Device &device);

    ~Array();

    Array(Array &&);
    Array(const Array &) = delete;

    Array &operator=(Array &&);
    Array &operator=(const Array &) = delete;

    size_t size() const { return size_; }
    size_t nElem() const { return nElem_; }
    DataType dtype() const { return dtype_; }

    void fromCPU(const void *other, size_t size);
    void toCPU(void *other, size_t size);

    void *raw() const { return ptr_; }
};

} // namespace ir

#endif // ARRAY_H
