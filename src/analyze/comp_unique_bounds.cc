#include <analyze/all_uses.h>
#include <analyze/comp_unique_bounds.h>

namespace freetensor {

namespace {

class CountHeavyOps : public Visitor {
    int cnt_ = 0;

  public:
    int cnt() const { return cnt_; }

  protected:
    void visitExpr(const Expr &op) {
        Visitor::visitExpr(op);
        if (!op->isConst() && op->nodeType() != ASTNodeType::Add &&
            op->nodeType() != ASTNodeType::Sub &&
            op->nodeType() != ASTNodeType::Mul) {
            cnt_++;
        }
    }
};

} // Anonymous namespace

int CompUniqueBounds::Bound::countHeavyOps(const Expr &op) {
    CountHeavyOps visitor;
    visitor(op);
    return visitor.cnt();
}

int CompUniqueBounds::Bound::countScope(
    const Expr &expr,
    const std::unordered_map<std::string, int> &orderedScope) {
    int scope = -1; // 0 = first level var, -1 = no var
    for (auto &&use : allNames(expr)) {
        scope = std::max(scope, orderedScope.at(use));
    }
    return scope;
}

} // namespace freetensor
