#ifndef FREE_TENSOR_MAKE_1D_VAR_H
#define FREE_TENSOR_MAKE_1D_VAR_H

#include <unordered_map>

#include <itertools.hpp>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

/**
 * Some backends do not support dynamic array types, so we transform them into
 * 1D arrays, so they can be passed by pointers
 */
class Make1DVar : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    template <class T> T visitMemAcc(const T &op) {
        auto &&shape = buffer(op->var_)->tensor()->shape();
        if (shape.size() <= 1) {
            return op;
        }
        auto &&indices = op->indices_;
        size_t ndim = indices.size();
        ASSERT(shape.size() == ndim);

        Expr res;
        for (auto &&[dim, idx] : iter::zip(shape, indices)) {
            res = res.isValid() ? makeMul(res, dim) : res;
            res = res.isValid() ? makeAdd(res, idx) : (Expr)idx;
        }
        op->indices_ = {res};
        return op;
    }

  protected:
    using BaseClass::visit;

    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

Stmt make1dVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(make1dVar)

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_1D_VAR_H
