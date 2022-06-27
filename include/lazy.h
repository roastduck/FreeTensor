#ifndef FREE_TENSOR_LAZY_H
#define FREE_TENSOR_LAZY_H

#include <functional>
#include <type_traits>

template <typename T> class Lazy {
    std::optional<T> container_;
    std::function<T()> delayedInit_;

  public:
    const T &operator*() {
        if (!container_)
            container_ = delayedInit_();
        return *container_;
    }

    template <typename F>
    Lazy(F delayedInit) : container_(std::nullopt), delayedInit_(delayedInit) {}
};

template <typename F>
Lazy(F delayedInit) -> Lazy<std::decay_t<decltype(std::declval<F>()())>>;

#define LAZY(expr) (Lazy([&]() { return (expr); }))

#endif // FREE_TENSOR_LAZY_H
