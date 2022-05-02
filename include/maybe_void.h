#ifndef FREE_TENSOR_MAYBE_VOID_H
#define FREE_TENSOR_MAYBE_VOID_H

#include <type_traits>
#include <variant>

namespace freetensor {

template <class T> struct TypeOrMonostate { typedef T type; };
template <> struct TypeOrMonostate<void> { typedef std::monostate type; };

/**
 * Expand to `name = expr` for normal types, and expand to `expr` for void
 */
#define MAYBE_VOID_ASSIGN(name, expr)                                          \
    if constexpr (std::is_same_v<decltype(expr), void>) {                      \
        expr;                                                                  \
    } else {                                                                   \
        name = expr;                                                           \
    }

/**
 * Expand to `auto name = expr` for normal types, and expand to `expr` for void
 */
#define MAYBE_VOID(name, expr)                                                 \
    [[maybe_unused]] typename TypeOrMonostate<decltype(expr)>::type name;      \
    MAYBE_VOID_ASSIGN(name, expr)

} // namespace freetensor

#endif // FREE_TENSOR_MAYBE_VOID_H
