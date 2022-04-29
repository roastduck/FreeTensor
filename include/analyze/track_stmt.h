#ifndef FREE_TENSOR_TRACK_STMT_H
#define FREE_TENSOR_TRACK_STMT_H

#include <vector>

#include <maybe_void.h>
#include <stmt.h>

namespace freetensor {

/**
 * A context for Visitor or Mutator that tracks the visiting statements stack
 *
 * Inherit this class to use. E.g., inherit TrackStmt<Visitor> or
 * TrackStmt<Mutator>
 *
 * This class will automatically maintains the statement stack if the sub-class
 * calls TrackStmt::visitStmt, which is the suggested usage
 *
 * However, in some cases, this is impossible, e.g., when the sub-class needs to
 * recurse into different sub-trees manually. In these cases, the sub-class
 * should explicitly call the pushStmt / popStmt methods
 */
template <class BaseClass> class TrackStmt : public BaseClass {
    std::vector<Stmt> stmtStack_;

  protected:
    void pushStmt(const Stmt &op) { stmtStack_.emplace_back(op); }
    void popStmt(const Stmt &) { stmtStack_.pop_back(); }

    const Stmt &curStmt() const {
        ASSERT(!stmtStack_.empty());
        return stmtStack_.back();
    }

    typename BaseClass::StmtRetType visitStmt(const Stmt &op) override {
        pushStmt(op);
        MAYBE_VOID(ret, BaseClass::visitStmt(op));
        popStmt(op);
        if constexpr (!std::is_same_v<typename BaseClass::StmtRetType, void>) {
            return ret;
        }
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_TRACK_STMT_H
