#ifndef FREE_TENSOR_USER_GRAD_H
#define FREE_TENSOR_USER_GRAD_H

#include <optional>
#include <unordered_set>

#include <stmt.h>

namespace freetensor {

struct StmtSetToUserGrad {
    /// Set of statements in the original program
    std::unordered_set<ID> oriStmts_;

    /// Backward statement (can be a scope)
    Stmt bwdBody_;

    StmtSetToUserGrad(const std::unordered_set<ID> &oriStmts,
                      const Stmt &bwdBody)
        : oriStmts_(oriStmts), bwdBody_(bwdBody) {}
};

struct RangeToUserGrad {
    /// Range of statements in the original program, inclusive
    ID oriBegin_, oriEnd_;

    /// Backward statement (can be a scope)
    Stmt bwdBody_;

    RangeToUserGrad(const ID &oriBegin, const ID &oriEnd, const Stmt &bwdBody)
        : oriBegin_(oriBegin), oriEnd_(oriEnd), bwdBody_(bwdBody) {}
};

std::optional<std::pair<ID, ID>>
getRangeFromStmtSeq(const Stmt &op, const std::unordered_set<ID> &stmts);

} // namespace freetensor

#endif // FREE_TENSOR_USER_GRAD_H
