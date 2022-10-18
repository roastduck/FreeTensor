#ifndef FREE_TENSOR_REPLACE_ITER_H
#define FREE_TENSOR_REPLACE_ITER_H

#include <unordered_map>

#include <hash.h>
#include <mutator.h>

namespace freetensor {

/**
 * Replace all Var node with a specific name by another expression
 */
class ReplaceIter : public Mutator {
    std::unordered_map<std::string, Expr> replace_;

  public:
    ReplaceIter(const std::string &name, const Expr &expr)
        : replace_({{name, expr}}) {}
    ReplaceIter(const std::unordered_map<std::string, Expr> &replace)
        : replace_(replace) {}

  protected:
    Expr visit(const Var &op) override {
        if (auto it = replace_.find(op->name_); it != replace_.end()) {
            return it->second;
        } else {
            return op;
        }
    }
};

class ReplaceIterAndRecordLog : public Mutator {
    std::unordered_map<std::string, Expr> replace_;
    std::unordered_map<StmtOrExprID, Expr> &replacedLog_;

  public:
    ReplaceIterAndRecordLog(
        const std::unordered_map<std::string, Expr> &replace,
        std::unordered_map<StmtOrExprID, Expr> &replacedLog)
        : replace_(replace), replacedLog_(replacedLog) {}

  protected:
    Expr visitExpr(const Expr &expr) override {
        auto newExpr = Mutator::visitExpr(expr);
        if (!HashComparator{}(expr, newExpr)) {
            replacedLog_[{expr, expr->parentStmt()}] = newExpr;
        }
        return newExpr;
    }

    Expr visit(const Var &op) override {
        if (auto it = replace_.find(op->name_); it != replace_.end()) {
            return it->second;
        } else {
            return op;
        }
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_REPLACE_ITER_H
