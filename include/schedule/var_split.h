#ifndef FREE_TENSOR_VAR_SPLIT_H
#define FREE_TENSOR_VAR_SPLIT_H

#include <mutator.h>

namespace freetensor {

enum VarSplitMode : int { FixedSize, RelaxedSize };

class VarSplit : public Mutator {
    ID def_;
    std::string var_;
    int dim_;
    bool fixedSize_;
    int factor_, nparts_;
    Expr dynFactor_;
    bool found_ = false;

  public:
    VarSplit(const ID &def, int dim, bool fixedSize, int factor, int nparts)
        : def_(def), dim_(dim), fixedSize_(fixedSize), factor_(factor),
          nparts_(nparts) {}

    bool found() const { return found_; }

  private:
    template <class T> T splitMemAcc(const T &op) {
        if (op->var_ == var_) {
            Expr x = op->indices_[dim_];
            op->indices_[dim_] = makeFloorDiv(x, dynFactor_);
            op->indices_.insert(op->indices_.begin() + dim_ + 1,
                                makeMod(x, dynFactor_));
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

Stmt varSplit(const Stmt &ast, const ID &def, int dim, VarSplitMode mode,
              int factor, int nparts);

} // namespace freetensor

#endif // FREE_TENSOR_VAR_SPLIT_H
