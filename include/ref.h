#ifndef REF_H
#define REF_H

#include <functional> // hash
#include <memory>
#include <type_traits>

#include <allocator.h>
#include <except.h>

namespace ir {

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

    std::shared_ptr<T> ptr_;

  private:
    Ref(std::shared_ptr<T> &&ptr) : ptr_(ptr) {}

  public:
    typedef T Object;

    Ref() = default;
    Ref(std::nullptr_t) : Ref() {}

    /// NO NOT USE THIS CONSTRUCTOR IN PUBLIC
    /// It is public because Pybind11 needs it
    Ref(T *ptr) : ptr_(ptr) {}

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

    Ref clone() const {
        return Ref(std::allocate_shared<T>(Allocator<T>(), *ptr_));
    }

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
        ASSERT(isValid());
        return ptr_.get();
    }

    static Ref make() { return Ref(std::allocate_shared<T>(Allocator<T>())); }
    static Ref make(T &&x) {
        return Ref(std::allocate_shared<T>(Allocator<T>(), std::move(x)));
    }
    static Ref make(const T &x) {
        return Ref(std::allocate_shared<T>(Allocator<T>(), x));
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

} // namespace ir

namespace std {

template <class T> struct hash<ir::Ref<T>> {
    hash<T *> hash_;
    size_t operator()(const ir::Ref<T> &ref) const { return hash_(ref.get()); }
};

} // namespace std

#endif // REF_H
