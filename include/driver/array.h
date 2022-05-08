#ifndef FREE_TENSOR_ARRAY_H
#define FREE_TENSOR_ARRAY_H

#include <cstdint>
#include <vector>

#include <driver/device.h>
#include <tensor.h>

namespace freetensor {

/**
 * Data stored on a `Device` or shared by multiple `Device`s
 */
class Array {
    std::vector<std::pair<Ref<Device>, uint8_t *>> ptrs_;
    size_t size_ = 0, nElem_ = 0;
    std::vector<size_t> shape_;
    DataType dtype_;
    Ref<Device> preferDevice_;

  public:
    /**
     * Intialize an array on a specific device
     *
     * @param shape : Length of each dimensions of the array
     * @param dtype : Data type of the array
     * @param preferDevice : Store the data at this device by default. If
     * omitted, use the default one in Config
     * @{
     */
    Array(const std::vector<size_t> &shape, DataType dtype);
    Array(const std::vector<size_t> &shape, DataType dtype,
          const Ref<Device> &preferDevice);
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

    void *rawSharedTo(const Ref<Device> &device);
    void *rawMovedTo(const Ref<Device> &device);
    void *rawInitTo(const Ref<Device> &device);

    void fromCPU(const void *other, size_t size);
    void toCPU(void *other, size_t size);

    Ref<Device> preferDevice() const { return preferDevice_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_ARRAY_H
