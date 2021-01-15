#ifndef ARRAY_H
#define ARRAY_H

#include <cstdint>
#include <string>
#include <vector>

#include <driver/device.h>
#include <tensor.h>

namespace ir {

class Array {
    uint8_t *ptr_ = nullptr;
    size_t size_ = 0;

    DataType dtype_;
    std::vector<size_t> shape_;
    Device device_;

  public:
    Array(const std::vector<size_t> &shape, DataType dtype,
          const Device &device);
    ~Array();

    Array(Array &&);
    Array(const Array &) = delete;

    Array &operator=(Array &&);
    Array &operator=(const Array &) = delete;

    size_t size() const { return size_; }
    DataType dtype() const { return dtype_; }
    const std::vector<size_t> &shape() const { return shape_; }

    void fromCPU(const void *other, size_t size);
    void toCPU(void *other, size_t size);

    void *raw() const { return ptr_; }
};

} // namespace ir

#endif // ARRAY_H
