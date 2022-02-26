#ifndef OPT_H
#define OPT_H

#include <optional>

namespace ir {

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

} // namespace ir

#endif // OPT_H
