#ifndef AS_DNF_H
#define AS_DNF_H

#include <visitor.h>

namespace ir {

typedef std::vector<std::vector<Expr>> DNF;

class AsDNF : public Visitor {
    std::unordered_map<Expr, DNF> results_;
    bool neg_ = false;

  public:
    const DNF &results(const Expr &root) const { return results_.at(root); }

  protected:
    void visitExpr(const Expr &op) override;
    void visit(const LAnd &op) override;
    void visit(const LOr &op) override;
    void visit(const LNot &op) override;
    void visit(const EQ &op) override;
    void visit(const NE &op) override;
    void visit(const LE &op) override;
    void visit(const LT &op) override;
    void visit(const GE &op) override;
    void visit(const GT &op) override;
};

DNF asDNF(const Expr &expr);

} // namespace ir

#endif // AS_DNF_H
