#ifndef VAR_SPLIT_H
#define VAR_SPLIT_H

#include <mutator.h>

namespace ir {

class VarSplit : public Mutator {
    std::string def_, var_;
    int dim_;
    bool fixedSize_;
    int factor_, nparts_;
    Expr dynFactor_;
    bool found_ = false;

  public:
    VarSplit(const std::string &def, int dim, bool fixedSize, int factor,
             int nparts)
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

} // namespace ir

#endif // VAR_SPLIT_H
