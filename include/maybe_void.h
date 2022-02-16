#ifndef MAYBE_VOID_H
#define MAYBE_VOID_H

#include <type_traits>
#include <variant>

namespace ir {

template <class T> struct TypeOrMonostate { typedef T type; };
template <> struct TypeOrMonostate<void> { typedef std::monostate type; };

/**
 * Expand to `auto name = expr` for normal types, and expand to `expr` for void
 */
#define MAYBE_VOID(name, expr)                                                 \
    [[maybe_unused]] typename TypeOrMonostate<decltype(expr)>::type name;      \
    if constexpr (std::is_same_v<decltype(expr), void>) {                      \
        expr;                                                                  \
    } else {                                                                   \
        name = expr;                                                           \
    }

} // namespace ir

#endif // MAYBE_VOID_H
