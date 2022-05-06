#ifndef FREE_TENSOR_OPT_H
#define FREE_TENSOR_OPT_H

#include <optional>

namespace freetensor {

template <class T> class Opt {
    std::optional<T> opt_;

  private:
    Opt(std::optional<T> &&opt) : opt_(std::move(opt)) {}

  public:
    Opt() = default;
    Opt(std::nullptr_t) : Opt() {}

    bool isValid() const { return opt_.has_value(); }

    T &operator*() const {
        ASSERT(isValid());
        return *opt_;
    }

    T *operator->() const {
        ASSERT(isValid());
        return &*opt_;
    }

    T &operator*() {
        ASSERT(isValid());
        return *opt_;
    }

    T *operator->() {
        ASSERT(isValid());
        return &*opt_;
    }

    static Opt make() { return Opt(T()); }
    static Opt make(T &&x) { return Opt(std::move(x)); }
    static Opt make(const T &x) { return Opt(x); }
};

} // namespace freetensor

#endif // FREE_TENSOR_OPT_H
