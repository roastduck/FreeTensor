#ifndef FREE_TENSOR_SPLIT_H
#define FREE_TENSOR_SPLIT_H

#include <string>

#include <mutator.h>

namespace freetensor {

class Splitter : public Mutator {
    ID src_, dst0_, dst1_;
    int factor_ = -1, nparts_ = -1;

    std::string iterFrom_;
    Expr iterTo_;

    bool found_ = false;

  public:
    Splitter(const ID &id, int factor = -1, int nparts = -1)
        : src_(id), dst0_(id.strId() + ".0"), dst1_(id.strId() + ".1"),
          factor_(factor), nparts_(nparts) {}

    const ID &outerId() const { return dst0_; }
    const ID &innerId() const { return dst1_; }
    bool found() const { return found_; }

  protected:
    Stmt visit(const For &op) override;
    Expr visit(const Var &op) override;
};

std::pair<Stmt, std::pair<ID, ID>> split(const Stmt &ast, const ID &id,
                                         int factor, int nparts);

} // namespace freetensor

#endif // FREE_TENSOR_SPLIT_H
