#ifndef REMOVE_WRITES_H
#define REMOVE_WRITES_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindLoopInvariantWrites : public Visitor {
    std::vector<For> loopStack_;
    std::vector<If> ifStack_;
    std::vector<std::pair<Store, Expr>> results_; /// (store, extraCond)
    std::unordered_map<std::string, int> defDepth_;
    const std::unordered_map<Expr, std::unordered_set<std::string>>
        &variantExpr_;

  public:
    FindLoopInvariantWrites(
        const std::unordered_map<Expr, std::unordered_set<std::string>>
            &variantExpr)
        : variantExpr_(variantExpr) {}

    const std::vector<std::pair<Store, Expr>> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const VarDef &op) override;
    void visit(const Store &op) override;
};

class RemoveWrites : public Mutator {
    const std::unordered_set<Stmt> &redundant_;
    const std::unordered_map<Stmt, Stmt> &replacement_;

  public:
    RemoveWrites(const std::unordered_set<Stmt> &redundant,
                 const std::unordered_map<Stmt, Stmt> &replacement)
        : redundant_(redundant), replacement_(replacement) {}

    template <class T> Stmt doVisit(const T &op) {
        if (redundant_.count(op)) {
            return makeStmtSeq(op->id(), {});
        } else if (replacement_.count(op)) {
            return replacement_.at(op);
        } else {
            return Mutator::visit(op);
        }
    }

  protected:
    Stmt visit(const Store &op) override { return doVisit(op); }
    Stmt visit(const ReduceTo &op) override { return doVisit(op); }
};

/**
 * Remove two types of redundant writes
 *
 * Type 1: Transform
 *
 * ```
 * x[0] = 1;
 * x[1] = 2;
 * ```
 *
 * to
 *
 * ```
 * x[1] = 2;
 * ```
 *
 * Type 2: Transform
 *
 * ```
 * for (i = 0; i < 5; i++) {
 *   x[i] = i;
 * }
 * ```
 *
 * to
 *
 * ```
 * for (i = 0; i < 5; i++) {
 *   if (i == 4) {
 *      x[i] = i;
 *   }
 * }
 * ```
 */
Stmt removeWrites(const Stmt &op);

} // namespace ir

#endif // REMOVE_WRITES_H
