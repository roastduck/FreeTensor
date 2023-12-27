#ifndef FREE_TENSOR_CHECK_LOOP_ORDER_H
#define FREE_TENSOR_CHECK_LOOP_ORDER_H

#include <string>
#include <unordered_set>
#include <vector>

#include <visitor.h>

namespace freetensor {

/**
 * Return loops in nesting order
 *
 * @param requireRangeDefinedOutside : If true, throw InvalidSchedule when
 * the range of any loop is an expression of the iterating variable of other
 * loops in the nest. For example, `for i = 0 to 4 { for j = 0 to i {}}`
 * will be illegal.
 */
class CheckLoopOrder : public Visitor {
    std::vector<ID> dstOrder_;
    std::vector<For> curOrder_, outerLoops_, outerLoopStack_;
    std::vector<StmtSeq> stmtSeqStack_, stmtSeqInBetween_;
    std::unordered_set<std::string> itersDefinedInNest_;
    bool reqireRangeDefinedOutside_;
    bool done_ = false;

  public:
    CheckLoopOrder(const std::vector<ID> &dstOrder,
                   bool reqireRangeDefinedOutside = true)
        : dstOrder_(dstOrder),
          reqireRangeDefinedOutside_(reqireRangeDefinedOutside) {}

    /**
     * All required loops, sorted from outer to inner
     */
    const std::vector<For> &order() const;

    /**
     * All StmtSeq nodes nested inside the outer-most required loop, and nesting
     * the inner-most required loop, sorted from outer to inner
     */
    const std::vector<StmtSeq> &stmtSeqInBetween() const {
        return stmtSeqInBetween_;
    }

    /**
     * Loops surrounding all loops in `dstOrder`
     */
    const std::vector<For> &outerLoops() const { return outerLoops_; }

  protected:
    void visitStmt(const Stmt &stmt) override;
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_CHECK_LOOP_ORDER_H
