#ifndef FREE_TENSOR_LAZY_H
#define FREE_TENSOR_LAZY_H

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>

namespace freetensor {

template <typename T> class Lazy {
    struct LazyData {
        std::optional<T> container_;
        std::function<T()> delayedInit_;
        std::once_flag onceFlag_;
    };

    // Although we deleted copy constructor and copy assignment operator, we
    // still want to use the move constructor and move assignment operator, but
    // std::once_flag is not movable. So we use std::unique_ptr to store
    // LazyData.
    std::unique_ptr<LazyData> data_ = std::make_unique<LazyData>();

  public:
    const T &operator*() {
        std::call_once(data_->onceFlag_,
                       [&] { data_->container_ = data_->delayedInit_(); });
        return data_->container_.value();
    }

    const T *operator->() { return &(this->operator*()); }

    template <typename F> Lazy(F delayedInit) {
        data_->delayedInit_ = delayedInit;
    }

    // We want to keep only one copy of LazyData. Otherwise, LazyData might be
    // re-initialized multiple times.
    //
    // A bad example:
    //
    // ```
    // a = Lazy(...)
    // for (...) { auto b = a; *b; } // re-initialized
    // ```
    //
    // A good example:
    //
    // ```
    // a = Lazy(...)
    // for (...) { auto &b = a; *b; } // initialized only once
    // ```
    Lazy(const Lazy &) = delete;
    Lazy &operator=(const Lazy &) = delete;
};

template <typename F>
Lazy(F delayedInit) -> Lazy<std::decay_t<decltype(std::declval<F>()())>>;

#define LAZY(expr) (Lazy([&]() { return (expr); }))

} // namespace freetensor

#endif // FREE_TENSOR_LAZY_H
