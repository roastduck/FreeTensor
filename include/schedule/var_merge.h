#ifndef VAR_MERGE_H
#define VAR_MERGE_H

#include <mutator.h>

namespace ir {

class VarMerge : public Mutator {
    ID def_;
    std::string var_;
    int dim_;
    Expr factor_;
    bool found_ = false;

  public:
    VarMerge(const ID &def, int dim) : def_(def), dim_(dim) {}

    bool found() const { return found_; }

  private:
    template <class T> T mergeMemAcc(const T &op) {
        if (op->var_ == var_) {
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

} // namespace ir

#endif // VAR_MERGE_H
