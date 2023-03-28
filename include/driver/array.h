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
 * on other devices will be dropped. Writing to a lending user object at a
 * different device is not supported.
 *
 * When an `Array` is requried for read-write, an `ArrayCopy` will be copied to
 * a specific device, if there isn't one, and copies on other devices will be
 * dropped. Writing to a lending user object at a different device is not
 * supported.
 */
class Array {
    // Identical data on different devices.
    std::vector<ArrayCopy> ptrs_;

    // Temorary copy of the data used by AccessType::InputMutable. The data may
    // become different after the program runs.
    std::optional<ArrayCopy> tempPtr_;

    size_t size_ = 0, nElem_ = 0;
    std::vector<size_t> shape_;
    DataType dtype_;
    bool dontDropBorrow_, moved_;

  private:
    /**
     * Intialize an array on a specific device
     *
     * @param shape : Length of each dimensions of the array
     * @param dtype : Data type of the array
     * @param dontDropBorrow : If true, report an error if we have to drop a
     * borrwed ArrayCopy. This flag can be set to true when the Array is
     * cunstructed IMPLICITLY from a user object by borrowing from it, where
     * users may expect they are acutually manipulating their user object,
     * instead of this Array
     * @param moved : If true, it means we do not care about data in this Array
     * any more after the program runs. Variables with "input-mutable" access
     * type may modify the Array
     */
    Array(const std::vector<size_t> &shape, DataType dtype,
          bool dontDropBorrow = false, bool moved = false);

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
                               DataType dtype, const Ref<Device> &device,
                               bool dontDropBorrow, bool moved);

    ~Array();

    Array(Array &&);
    Array(const Array &) = delete;

    Array &operator=(Array &&);
    Array &operator=(const Array &) = delete;

    size_t size() const { return size_; }
    size_t nElem() const { return nElem_; }
    const std::vector<size_t> &shape() const { return shape_; }
    DataType dtype() const { return dtype_; }

    bool dontDropBorrow() const { return dontDropBorrow_; }
    void setDontDropBorrow(bool flag) { dontDropBorrow_ = flag; }

    bool moved() const { return moved_; }
    void setMoved(bool flag) { moved_ = flag; }

    void *rawSharedTo(const Ref<Device> &device);
    void *rawMovedTo(const Ref<Device> &device);
    void *rawInitTo(const Ref<Device> &device);
    void *rawTemporarilyCopiedTo(const Ref<Device> &device);

    /**
     * Somethings we can't keep track of user objects, so we need to ensure we
     * don't borrow from user data
     */
    void makePrivateCopy();
};

} // namespace freetensor

#endif // FREE_TENSOR_ARRAY_H
