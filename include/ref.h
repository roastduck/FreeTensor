#ifndef REF_H
#define REF_H

#include <functional> // hash
#include <memory>

#include <except.h>

namespace ir {

/**
 * Ref-counting pointer
 */
template <class T> class Ref {
    template <class U> friend class Ref;

    std::shared_ptr<T> ptr_;

  public:
    typedef T Object;

    Ref() = default;
    Ref(std::nullptr_t) : Ref() {}

  private:
    Ref(T *ptr) : ptr_(ptr) {}

  public:
    /**
     * Shared with any compatible references
     */
    template <class U>
    Ref(const Ref<U> &other) : ptr_(std::static_pointer_cast<T>(other.ptr_)) {}

    template <class U> Ref &operator=(const Ref<U> &other) {
        ptr_ = std::static_pointer_cast<T>(other.ptr_);
        return *this;
    }

    Ref clone() const { return Ref(new T(*ptr_)); }

    template <class U> Ref<U> as() const { return Ref<U>(*this); }

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

    static Ref make() { return Ref(new T()); }
    static Ref make(T &&x) { return Ref(new T(std::move(x))); }
    static Ref make(const T &x) { return Ref(new T(x)); }

    friend bool operator==(const Ref &lhs, const Ref &rhs) {
        return lhs.ptr_ == rhs.ptr_;
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
