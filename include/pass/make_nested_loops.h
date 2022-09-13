#ifndef FREE_TENSOR_MAKE_NESTED_LOOPS_H
#define FREE_TENSOR_MAKE_NESTED_LOOPS_H

#include <type_traits>

#include <container_utils.h>
#include <stmt.h>

namespace freetensor {

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
    for (auto &&[_iter, begin, _end, step, _len, property] : views::reverse(
             views::zip(iters, begins, ends, steps, lens, properties))) {
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
        auto &&end = ((Expr)_end).isValid()
                         ? (Expr)_end
                         : makeAdd(begin, makeMul(_len, step));
        auto &&len = ((Expr)_len).isValid()
                         ? (Expr)_len
                         : makeCeilDiv(makeSub(_end, begin), step);
        ret = makeFor(*iter, begin, end, step, len, property, ret);
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_NESTED_LOOPS_H
