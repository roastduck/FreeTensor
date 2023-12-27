#ifndef FREE_TENSOR_SYNC_FUNC_H
#define FREE_TENSOR_SYNC_FUNC_H

#include <functional>
#include <mutex>

namespace freetensor {

namespace detail {

template <class T> struct TaggedSyncFunc { T f_; };
template <class T> struct TaggedUnsyncFunc { T f_; };

} // namespace detail

template <typename R, typename... Params> class SyncFunc;

/**
 * A wrapped function that automatically can lock itself or not when called
 */
template <typename R, typename... Params> class SyncFunc<R(Params...)> {
    std::function<R(Params...)> f_;
    bool synchronized_ = true;
    mutable std::mutex
        mutex_; // mutable to allow const callback object to lock against

  public:
    SyncFunc() {}
    SyncFunc(std::nullptr_t) {}

    explicit SyncFunc(const std::function<R(Params...)> &f, bool sync)
        : f_(f), synchronized_(sync) {}
    explicit SyncFunc(std::function<R(Params...)> &&f, bool sync)
        : f_(std::move(f)), synchronized_(sync) {}

    template <class T>
    SyncFunc(const detail::TaggedSyncFunc<T> &f) : SyncFunc(f.f_, true) {}
    template <class T>
    SyncFunc(detail::TaggedSyncFunc<T> &&f) : SyncFunc(std::move(f.f_), true) {}

    template <class T>
    SyncFunc(const detail::TaggedUnsyncFunc<T> &f) : SyncFunc(f.f_, false) {}
    template <class T>
    SyncFunc(detail::TaggedUnsyncFunc<T> &&f)
        : SyncFunc(std::move(f.f_), false) {}

    SyncFunc(const SyncFunc &other)
        : f_(other.f_), synchronized_(other.synchronized_), mutex_() {}
    SyncFunc &operator=(const SyncFunc &other) {
        f_ = other.f_;
        synchronized_ = other.synchronized_;
        return *this;
    }

    friend bool operator==(const SyncFunc &self, std::nullptr_t) {
        return self.f_ == nullptr;
    }
    friend bool operator==(std::nullptr_t, const SyncFunc &self) {
        return self.f_ == nullptr;
    }

    template <typename... Args> R operator()(Args &&...args) const {
        if (synchronized_) {
            std::lock_guard lg(mutex_);
            return f_(std::forward<Args>(args)...);
        } else {
            return f_(std::forward<Args>(args)...);
        }
    }
};

/**
 * Wrap a function to automatically lock itself when called
 */
template <class T>
detail::TaggedSyncFunc<std::remove_reference_t<T>> syncFunc(T &&f) {
    return {std::forward<T>(f)};
}

/**
 * Make explicit that a function is not locked when called concurrently
 */
template <class T>
detail::TaggedUnsyncFunc<std::remove_reference_t<T>> unsyncFunc(T &&f) {
    return {std::forward<T>(f)};
}

} // namespace freetensor

#endif // FREE_TENSOR_SYNC_FUNC_H
