#ifndef REMOVE_WRITES_H
#define REMOVE_WRITES_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/find_loop_variance.h>
#include <analyze/symbol_table.h>
#include <analyze/with_cursor.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindLoopInvariantWrites : public SymbolTable<WithCursor<Visitor>> {
    typedef SymbolTable<WithCursor<Visitor>> BaseClass;

    std::vector<Cursor> cursorStack_;
    std::vector<If> ifStack_;
    std::unordered_map<Store, std::tuple<VarDef, Expr, Cursor>>
        results_; /// (store, extraCond, cursor to loop)
    std::unordered_map<std::string, int> defDepth_;
    const std::unordered_map<
        Expr, std::unordered_map<std::string, LoopVariability>> &variantExpr_;
    const std::string &singleDefId_;

  public:
    FindLoopInvariantWrites(
        const std::unordered_map<
            Expr, std::unordered_map<std::string, LoopVariability>>
            &variantExpr,
        const std::string &singleDefId)
        : variantExpr_(variantExpr), singleDefId_(singleDefId) {}

    const std::unordered_map<Store, std::tuple<VarDef, Expr, Cursor>> &
    results() const {
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
            auto ret = replacement_.at(op);
            ret->setId(op->id());
            return ret;
        } else {
            return Mutator::visit(op);
        }
    }

  protected:
    Stmt visit(const Store &op) override { return doVisit(op); }
    Stmt visit(const ReduceTo &op) override { return doVisit(op); }
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
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
Stmt removeWrites(const Stmt &op, const std::string &singleDefId = "");

DEFINE_PASS_FOR_FUNC(removeWrites)

} // namespace ir

#endif // REMOVE_WRITES_H
