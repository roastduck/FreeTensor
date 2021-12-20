#ifndef MAKE_1D_VAR_H
#define MAKE_1D_VAR_H

#include <unordered_map>

#include <itertools.hpp>

#include <func.h>
#include <mutator.h>

namespace ir {

/**
 * Some backends do not support dynamic array types, so we transform them into
 * 1D arraies, so they can be passed by pointers
 */
class Make1DVar : public Mutator {
    std::unordered_map<std::string, Ref<Buffer>> buffers_;

  private:
    template <class T> T visitMemAcc(const T &op) {
        if (!buffers_.count(op->var_)) {
            return op;
        }
        auto &&shape = buffers_.at(op->var_)->tensor().shape();
        auto &&indices = op->indices_;
        size_t ndim = indices.size();
        ASSERT(shape.size() == ndim);

        Expr res;
        for (auto &&[dim, idx] : iter::zip(shape, indices)) {
            res = res.isValid() ? makeMul(res, dim) : res;
            res = res.isValid() ? makeAdd(res, idx) : (Expr)idx;
        }
        op->indices_ = std::vector<SubTree<ExprNode>>({res});
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

Stmt make1dVar(const Stmt &op);

DEFINE_PASS_FOR_FUNC(make1dVar)

} // namespace ir

#endif // MAKE_1D_VAR_H
