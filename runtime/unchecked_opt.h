#ifndef FREE_TENSOR_UNCHECKED_OPT_H
#define FREE_TENSOR_UNCHECKED_OPT_H

#include <optional>

#ifdef __CUDA_ARCH__
#define FUNC_ATTR __attribute__((always_inline)) __host__ __device__
#else
#define FUNC_ATTR __attribute__((always_inline))
#endif

/**
 * Unchecked `std::optional`-like holder
 *
 * It holds an object of type T, or null. Users are expected to know whether
 * there is an object. Only when there is really an object, it can be accessed
 *
 * If an object is currently holded and it has an non-trivial destructor,
 * `drop()` must be called before setting `UncheckedOpt` to another value, or
 * destructing `UncheckedOpt`
 *
 * `UncheckedOpt` is useful for holding an uninitialized object which does not
 * have a default constructor
 */
template <class T> class UncheckedOpt {
    union {
        T obj_;
        std::nullopt_t null_;
    };

  public:
    FUNC_ATTR UncheckedOpt() : null_(std::nullopt) {}

    /**
     * Setting the holded object
     *
     * If an object is currently holded and it has an non-trivial destructor,
     * `drop()` must be called before
     */
    FUNC_ATTR UncheckedOpt(const T &obj) : obj_(obj) {}
    FUNC_ATTR UncheckedOpt(T &&obj) : obj_(std::move(obj)) {}
    FUNC_ATTR UncheckedOpt(std::nullopt_t) : null_(std::nullopt) {}

    /**
     * Destructor
     *
     * If an object is currently holded and it has an non-trivial destructor,
     * `drop()` must be called before
     */
    ~UncheckedOpt() {}

    /**
     * Access the holded object
     *
     * These functions can only be called if there is really an object
     *
     * @{
     */
    FUNC_ATTR T &operator*() const { return obj_; }
    FUNC_ATTR T &operator*() { return obj_; }
    FUNC_ATTR T *operator->() const { return &obj_; }
    FUNC_ATTR T *operator->() { return &obj_; }
    /** @} */

    /**
     * Drop the holded object
     *
     * This functions can only be called if there is really an object
     */
    FUNC_ATTR void drop() { obj_.~T(); }
};

#endif // FREE_TENSOR_UNCHECKED_OPT_H
