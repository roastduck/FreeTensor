#ifndef FREE_TENSOR_ALL_USES_H
#define FREE_TENSOR_ALL_USES_H

#include <unordered_set>

#include <analyze/find_stmt.h>
#include <visitor.h>

namespace freetensor {

/**
 * Record all buffers that are used in an AST
 */
class AllUses : public Visitor {
  public:
    typedef int AllUsesType;
    static constexpr AllUsesType CHECK_LOAD = 0x1;
    static constexpr AllUsesType CHECK_STORE = 0x2;
    static constexpr AllUsesType CHECK_REDUCE = 0x4;
    static constexpr AllUsesType CHECK_VAR = 0x8;

  private:
    AllUsesType type_;
    bool noRecurseIdx_, noRecurseSubStmt_;
    bool inFirstStmt_ = false;
    std::unordered_set<std::string> uses_;

  public:
    AllUses(AllUsesType type, bool noRecurseIdx, bool noRecurseSubStmt)
        : type_(type), noRecurseIdx_(noRecurseIdx),
          noRecurseSubStmt_(noRecurseSubStmt) {}

    const std::unordered_set<std::string> &uses() const { return uses_; }

  protected:
    // TODO: Do we need to check viewOf, reduction, Alloc or Free?
    void visitStmt(const Stmt &s) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
};

/**
 * Find names of VarDef nodes or iterators of For nodes used in a specific type
 * of manner
 *
 * @param type : Filter how the name is used
 * @param filter : Filter statements to find. Can be a `fn(Stmt) -> bool` or a
 * `Ref<Selector>` or a selector string. If using `filter`, you must filter all
 * statements including those in the sub-trees (`noRecurseSubStmt` is always set
 * to `true`).
 * @param noRecurseIdx : If true, do not include names used as indirect indices
 * @param noRecurseSubStmt : If true, consider only the root statement(s)
 *
 * @{
 */
std::unordered_set<std::string>
allUses(const AST &op,
        AllUses::AllUsesType type = AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                                    AllUses::CHECK_REDUCE,
        bool noRecurseIdx = false, bool noRecurseSubStmt = false);
inline std::unordered_set<std::string>
allUses(const auto &filter, const Stmt &op,
        AllUses::AllUsesType type = AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                                    AllUses::CHECK_REDUCE,
        bool noRecurseIdx = false) {
    std::unordered_set<std::string> ret;
    for (auto &&stmt : findAllStmt(op, filter)) {
        for (auto &&use : allUses(stmt, type, noRecurseIdx, true)) {
            ret.emplace(use);
        }
    }
    return ret;
}
/** @} */

/**
 * Find names of all VarDef nodes that are read
 *
 * @{
 */
inline std::unordered_set<std::string> allReads(const AST &op,
                                                bool noRecurseIdx = false,
                                                bool noRecurseSubStmt = false) {
    return allUses(op, AllUses::CHECK_LOAD, noRecurseIdx, noRecurseSubStmt);
}
inline std::unordered_set<std::string>
allReads(const auto &filter, const Stmt &op, bool noRecurseIdx = false) {
    return allUses(filter, op, AllUses::CHECK_LOAD, noRecurseIdx);
}
/** @} */

/**
 * Find names of all VarDef nodes that are written
 *
 * @{
 */
inline std::unordered_set<std::string>
allWrites(const AST &op, bool noRecurseIdx = false,
          bool noRecurseSubStmt = false) {
    return allUses(op, AllUses::CHECK_STORE | AllUses::CHECK_REDUCE,
                   noRecurseIdx, noRecurseSubStmt);
}
inline std::unordered_set<std::string>
allWrites(const auto &filter, const Stmt &op, bool noRecurseIdx = false) {
    return allUses(filter, op, AllUses::CHECK_STORE | AllUses::CHECK_REDUCE,
                   noRecurseIdx);
}
/** @} */

/**
 * Find names of all iterators of For nodes that are used
 *
 * @{
 */
inline std::unordered_set<std::string> allIters(const AST &op,
                                                bool noRecurseIdx = false,
                                                bool noRecurseSubStmt = false) {
    return allUses(op, AllUses::CHECK_VAR, noRecurseIdx, noRecurseSubStmt);
}
inline std::unordered_set<std::string>
allIters(const auto &filter, const Stmt &op, bool noRecurseIdx = false) {
    return allUses(filter, op, AllUses::CHECK_VAR, noRecurseIdx);
}
/** @} */

/**
 * Find names of all VarDef nodes that are read or written, and all iterators of
 * For nodes that are used
 *
 * @{
 */
inline std::unordered_set<std::string> allNames(const AST &op,
                                                bool noRecurseIdx = false,
                                                bool noRecurseSubStmt = false) {
    return allUses(op,
                   AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                       AllUses::CHECK_REDUCE | AllUses::CHECK_VAR,
                   noRecurseIdx, noRecurseSubStmt);
}
inline std::unordered_set<std::string>
allNames(const auto &filter, const Stmt &op, bool noRecurseIdx = false) {
    return allUses(filter, op,
                   AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                       AllUses::CHECK_REDUCE | AllUses::CHECK_VAR,
                   noRecurseIdx);
}
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_ALL_USES_H
