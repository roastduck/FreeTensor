#ifndef FREE_TENSOR_ARRAY_H
#define FREE_TENSOR_ARRAY_H

#include <cstdint>
#include <vector>

#include <driver/device.h>
#include <tensor.h>

namespace freetensor {

/**
 * A copy of Array store on a specific device
 */
struct ArrayCopy {
    Ref<Device> device_;
    uint8_t *ptr_ = nullptr;
    bool borrowed_ = false;

    ArrayCopy(const Ref<Device> &device, uint8_t *ptr, bool borrowed)
        : device_(device), ptr_(ptr), borrowed_(borrowed) {}
};

/**
 * Data stored on a `Device` or shared by multiple `Device`s
 *
 * An `Array` manages one or more `ArrayCopy`s, locating on different devices.
 * An `ArrayCopy` can be a memory buffer managed by itself, or can be borrowed
 * from a user object.
 *
 * When an `Array` is required for read-only, an `ArrayCopy` will be copied to a
 * specific device, if there isn't one on the device.
 *
 * When an `Array` is requried for write-only, an `ArrayCopy` will be allocated
 * without initialization on a specific device, if there isn't one, and copies
 * on other devices will be dropped
 *
 * When an `Array` is requried for read-write, an `ArrayCopy` will be copied to
 * a specific device, if there isn't one, and copies on other devices will be
 * dropped
 */
class Array {
    std::vector<ArrayCopy> ptrs_;
    size_t size_ = 0, nElem_ = 0;
    std::vector<size_t> shape_;
    DataType dtype_;

  private:
    /**
     * Intialize an array on a specific device
     *
     * @param shape : Length of each dimensions of the array
     * @param dtype : Data type of the array
     * @param preferDevice : Store the data at this device by default. If
     * omitted, use the default one in Config
     */
    Array(const std::vector<size_t> &shape, DataType dtype);

  public:
    /**
     * Move from raw pointer. Use with cautious
     */
    static Array moveFromRaw(void *ptr, const std::vector<size_t> &shape,
                             DataType dtype, const Ref<Device> &device);

    /**
     * Borrow from raw pointer.
     *
     * Please make sure the owner keeps alive. Use `keep_alive` when exposing
     * with PyBind11
     */
    static Array borrowFromRaw(void *ptr, const std::vector<size_t> &shape,
                               DataType dtype, const Ref<Device> &device);

    ~Array();

    Array(Array &&);
    Array(const Array &) = delete;

    Array &operator=(Array &&);
    Array &operator=(const Array &) = delete;

    size_t size() const { return size_; }
    size_t nElem() const { return nElem_; }
    const std::vector<size_t> &shape() const { return shape_; }
    const std::vector<ArrayCopy> &ptrs() const { return ptrs_; }
    DataType dtype() const { return dtype_; }

    void *rawSharedTo(const Ref<Device> &device);
    void *rawMovedTo(const Ref<Device> &device);
    void *rawInitTo(const Ref<Device> &device);

    /**
     * Somethings we can't keep track of user objects, so we need to ensure we
     * don't borrow from user data
     */
    void makePrivateCopy();
};

} // namespace freetensor

#endif // FREE_TENSOR_ARRAY_H
