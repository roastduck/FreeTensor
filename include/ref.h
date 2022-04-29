#ifndef FREE_TENSOR_REF_H
#define FREE_TENSOR_REF_H

#include <functional> // hash
#include <memory>
#include <type_traits>

#include <allocator.h>
#include <except.h>

namespace freetensor {

class EnableSelfBase;
template <class T> class EnableSelf;

/**
 * Ref-counting pointer
 *
 * This class is thread-safe (For developers: concurrent accesses through
 * different `std::shared_ptr`s to the same object is already thread-safe, while
 * modifying the same `std::shared_ptr` is not. We never modify a `Ref`, so no
 * locks are needed. See https://en.cppreference.com/w/cpp/memory/shared_ptr)
 */
template <class T> class Ref {
    template <class U> friend class Ref;
    template <class U> friend class Weak;
    template <class U> friend class EnableSelf;

    std::shared_ptr<T> ptr_;

  private:
    Ref(std::shared_ptr<T> &&ptr) : ptr_(std::move(ptr)) { updateSelf(); }

    void updateSelf() {
        if constexpr (std::is_base_of_v<EnableSelfBase, T>) {
            if (ptr_ != nullptr) {
                std::static_pointer_cast<EnableSelf<typename T::Self>>(ptr_)
                    ->self_ = *this;
            }
        }
    }

  public:
    typedef T Object;

    Ref() = default;
    Ref(std::nullptr_t) : Ref() {}
    Ref(const Ref &) = default;
    Ref(Ref &&) = default;

    /// NO NOT USE THIS CONSTRUCTOR IN PUBLIC
    /// It is public because Pybind11 needs it
    Ref(T *ptr) : ptr_(ptr) { updateSelf(); }

    /**
     * Shared with any compatible references
     */
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    Ref(const Ref<U> &other) : ptr_(std::static_pointer_cast<T>(other.ptr_)) {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    Ref &operator=(const Ref<U> &other) {
        ptr_ = std::static_pointer_cast<T>(other.ptr_);
        return *this;
    }

    Ref &operator=(const Ref &) = default;
    Ref &operator=(Ref &&) = default;

    template <class U> Ref<U> as() const {
        Ref<U> ret;
        ret.ptr_ = std::static_pointer_cast<U>(ptr_);
        return ret;
    }

    bool isValid() const { return ptr_ != nullptr; }

    T &operator*() const {
        ASSERT(isValid());
        return *ptr_;
    }

    T *operator->() const {
        ASSERT(isValid());
        return ptr_.get();
    }

    T *get() const {
        return ptr_.get(); // maybe called from PyBind11, don't assert isValid()
    }

    static Ref make() { return Ref(std::allocate_shared<T>(Allocator<T>())); }
    static Ref make(T &&x) {
        return Ref(std::allocate_shared<T>(Allocator<T>(), std::move(x)));
    }
    static Ref make(const T &x) {
        return Ref(std::allocate_shared<T>(Allocator<T>(), x));
    }
    template <class... Args> static Ref make(Args &&...args) {
        return Ref(std::allocate_shared<T>(Allocator<T>(),
                                           std::forward<Args>(args)...));
    }

    friend bool operator==(const Ref &lhs, const Ref &rhs) {
        return lhs.ptr_ == rhs.ptr_;
    }
    friend bool operator!=(const Ref &lhs, const Ref &rhs) {
        return lhs.ptr_ != rhs.ptr_;
    }
    friend bool operator<(const Ref &lhs, const Ref &rhs) {
        return lhs.ptr_ < rhs.ptr_;
    }
};

template <class T> class Weak {
    std::weak_ptr<T> ptr_;
    bool notNull_ = false;

  public:
    Weak() {}
    Weak(std::nullptr_t) {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    Weak(const Ref<U> &ref) : ptr_(ref.ptr_), notNull_(ref.isValid()) {}

    /**
     * Return true if this is not a null pointer. If you are checking whether
     * the object it pointing to still exists, please use `lock().isValid()`
     */
    bool isValid() const { return notNull_; }

    Ref<T> lock() const { return Ref<T>(ptr_.lock()); }
};

class EnableSelfBase {};

/**
 * Similar to `std::enable_shared_from_this`, but returns a Ref
 *
 * Pybind11 has a bug with `std::enable_shared_from_this`
 * (https://github.com/pybind/pybind11/issues/3851), so we make our own
 * implementation, rather than wrap around `std::enable_shared_from_this`
 */
template <class T> class EnableSelf : public EnableSelfBase {
    template <class U> friend class Ref;

    Weak<T> self_;

  public:
    typedef T Self;

    Ref<T> self() const {
        auto ret = self_.lock();
        if (!ret.isValid()) {
            ERROR(
                "BUG: This class is not managed by Ref. Are you trying to get "
                "the Ref in a constructor even before a Ref is constructed?");
        }
        return ret;
    };
};

} // namespace freetensor

namespace std {

template <class T> struct hash<freetensor::Ref<T>> {
    hash<T *> hash_;
    size_t operator()(const freetensor::Ref<T> &ref) const {
        return hash_(ref.get());
    }
};

} // namespace std

#endif // FREE_TENSOR_REF_H
