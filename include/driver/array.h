#ifndef FREE_TENSOR_ARRAY_H
#define FREE_TENSOR_ARRAY_H

#include <cstdint>
#include <vector>

#include <driver/device.h>
#include <tensor.h>

namespace freetensor {

class Array {
    uint8_t *ptr_ = nullptr;
    size_t size_ = 0, nElem_ = 0;
    std::vector<size_t> shape_;
    DataType dtype_;
    Ref<Device> device_;

  public:
    /**
     * Intialize an array on a specific device
     *
     * @param shape : Length of each dimensions of the array
     * @param dtype : Data type of the array
     * @param device : Device that holds the array. If omitted, use the default
     * one set in Config
     * @{
     */
    Array(const std::vector<size_t> &shape, DataType dtype,
          const Ref<Device> &device);
    Array(const std::vector<size_t> &shape, DataType dtype);
    /** @} */

    /**
     * Move from raw pointer. Use with cautious
     */
    Array(void *ptr, const std::vector<size_t> &shape, DataType dtype,
          const Ref<Device> &device);

    ~Array();

    Array(Array &&);
    Array(const Array &) = delete;

    Array &operator=(Array &&);
    Array &operator=(const Array &) = delete;

    size_t size() const { return size_; }
    size_t nElem() const { return nElem_; }
    const std::vector<size_t> &shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    const Ref<Device> &device() const { return device_; }

    void fromCPU(const void *other, size_t size);
    void toCPU(void *other, size_t size);

    void *raw() const { return ptr_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_ARRAY_H
