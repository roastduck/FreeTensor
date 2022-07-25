#ifndef FUNC_UTILS_H
#define FUNC_UTILS_H

namespace freetensor {

/**
 * Convert a invocable from `f(x, y, ...)` to `f(*x, *y, ...)`
 */
template <class RefInvocable> class PtrInvocable {
    RefInvocable refInvocable_;

  public:
    template <class... Args> bool operator()(Args &&...args) {
        return refInvocable_(*args...);
    }
    template <class... Args> bool operator()(Args &&...args) const {
        return refInvocable_(*args...);
    }
};

} // namespace freetensor

#endif // FUNC_UTILS_H
