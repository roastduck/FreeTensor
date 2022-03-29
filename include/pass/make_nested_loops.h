#ifndef MAKE_NESTED_LOOPS_H
#define MAKE_NESTED_LOOPS_H

#include <type_traits>

#include <itertools.hpp>

#include <stmt.h>

namespace ir {

namespace detail {

// Hack on cppitertools so we can call reversed on repeat(...)
// https://github.com/ryanhaining/cppitertools/issues/86

template <class T>
const iter::impl::Repeater<T> &myrev(const iter::impl::Repeater<T> &repeater) {
    return repeater;
}

template <class T>
iter::impl::Repeater<T> &myrev(iter::impl::Repeater<T> &repeater) {
    return repeater;
}

template <class T> auto myrev(T &&x) {
    return iter::reversed(std::forward<T>(x));
}

} // namespace detail

/**
 * Helper function to make a loop nest given lists of loop parameters (from
 * outer to inner)
 *
 * Element of iters can be std::string, or Expr that can be casted to Var
 *
 * Pass nullptr ends or lens to set it automatically
 */
template <class Titers, class Tbegins, class Tends, class Tsteps, class Tlens,
          class Tproperties, class Tbody>
Stmt makeNestedLoops(Titers &&iters, Tbegins &&begins, Tends &&ends,
                     Tsteps &&steps, Tlens &&lens, Tproperties &&properties,
                     Tbody &&body) {
    Stmt ret = std::forward<Tbody>(body);
    for (auto &&[_iter, begin, _end, step, _len, property] :
         iter::zip(detail::myrev(iters), detail::myrev(begins),
                   detail::myrev(ends), detail::myrev(steps),
                   detail::myrev(lens), detail::myrev(properties))) {
        std::string *iter = nullptr;
        if constexpr (std::is_same_v<std::decay_t<decltype(_iter)>,
                                     std::string>) {
            iter = &_iter;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(_iter)>,
                                            Expr>) {
            ASSERT(_iter->nodeType() == ASTNodeType::Var);
            iter = &_iter.template as<VarNode>()->name_;
        } else {
            ASSERT(false);
        }
        auto &&end =
            ((Expr)_end).isValid() ? _end : makeAdd(begin, makeMul(_len, step));
        auto &&len = ((Expr)_len).isValid()
                         ? _len
                         : makeFloorDiv(makeSub(_end, begin), step);
        ret = makeFor("", *iter, begin, end, step, len, property, ret);
    }
    return ret;
}

} // namespace ir

#endif // MAKE_NESTED_LOOPS_H
