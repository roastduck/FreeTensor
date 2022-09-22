#ifndef FREE_TENSOR_VAR_MERGE_H
#define FREE_TENSOR_VAR_MERGE_H

#include <mutator.h>

namespace freetensor {

class VarMerge : public Mutator {
    ID def_;
    std::string var_, newVar_;
    int dim_;
    Expr factor_;
    bool found_ = false;

  public:
    VarMerge(const ID &def, int dim) : def_(def), dim_(dim) {}

    bool found() const { return found_; }

  private:
    template <class T> T mergeMemAcc(const T &op) {
        if (op->var_ == var_) {
            op->var_ = newVar_;
            Expr a = op->indices_[dim_], b = op->indices_[dim_ + 1];
            op->indices_[dim_] = makeAdd(makeMul(a, factor_), b);
            op->indices_.erase(op->indices_.begin() + dim_ + 1);
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

Stmt varMerge(const Stmt &ast, const ID &def, int dim);

} // namespace freetensor

#endif // FREE_TENSOR_VAR_MERGE_H
