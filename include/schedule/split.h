#ifndef SPLIT_H
#define SPLIT_H

#include <string>

#include <mutator.h>

namespace ir {

class Splitter : public Mutator {
    std::string src_, dst0_, dst1_;
    int factor_ = -1, nparts_ = -1;

    std::string iterFrom_;
    Expr iterTo_;

    bool found_ = false;

  public:
    Splitter(const std::string &id, int factor = -1, int nparts = -1)
        : src_(id), dst0_(id + ".0"), dst1_(id + ".1"), factor_(factor),
          nparts_(nparts) {}

    const std::string &outerId() const { return dst0_; }
    const std::string &innerId() const { return dst1_; }
    bool found() const { return found_; }

  protected:
    Stmt visit(const For &op) override;
    Expr visit(const Var &op) override;
};

std::pair<Stmt, std::pair<std::string, std::string>>
split(const Stmt &ast, const std::string &id, int factor, int nparts);

} // namespace ir

#endif // SPLIT_H
