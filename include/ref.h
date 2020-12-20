#ifndef REF_H
#define REF_H

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
    Ref() = default;

  private:
    Ref(T *ptr) : ptr_(ptr) {}

  public:
    Ref clone() const { return Ref(new T(*ptr_)); }

    /**
     * Shared with any compatible references
     */
    template <class U>
    Ref(const Ref<U> &other) : ptr_(std::static_pointer_cast<T>(other.ptr_)) {}

    template <class U> Ref &operator=(const Ref<U> &other) {
        ptr_ = std::static_pointer_cast<T>(other.ptr_);
        return *this;
    }

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
};

} // namespace ir

#endif // REF_H
