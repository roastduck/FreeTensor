#ifndef FREE_TENSOR_OPT_H
#define FREE_TENSOR_OPT_H

#include <optional>

namespace freetensor {

template <class T> class Opt {
    std::optional<T> opt_;

  public:
    Opt() = default;
    Opt(std::nullopt_t) : Opt() {}
    Opt(const T &item) : opt_(item) {}
    Opt(T &&item) : opt_(std::move(item)) {}
    Opt(const std::optional<T> &opt) : opt_(opt) {}
    Opt(std::optional<T> &&opt) : opt_(std::move(opt)) {}

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

    operator std::optional<T>() { return opt_; }

    static Opt make(T &&x) { return std::make_optional(std::move(x)); }
    static Opt make(const T &x) { return std::make_optional(x); }
    template <class... Args> static Opt make(Args &&...args) {
        return std::make_optional<T>(std::forward<Args>(args)...);
    }

    friend bool operator==(const Opt &lhs, const Opt &rhs) {
        return lhs.opt_ == rhs.opt_;
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_OPT_H
