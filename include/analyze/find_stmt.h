#ifndef FIND_STMT_H
#define FIND_STMT_H

#include <visitor.h>

namespace ir {

class FindStmtById : public Visitor {
    ID id_;
    Stmt result_;

  public:
    FindStmtById(const ID &id) : id_(id) {}

    const Stmt &result() const { return result_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline Stmt findStmt(const Stmt &ast, const ID &id) {
    FindStmtById visitor(id);
    visitor(ast);
    if (!visitor.result().isValid()) {
        throw InvalidSchedule("Statement " + toString(id) + " not found");
    }
    return visitor.result();
}

class FindStmtByFilter : public Visitor {
    const std::function<bool(const Stmt &)> &filter_;
    std::vector<Stmt> results_;

  public:
    FindStmtByFilter(const std::function<bool(const Stmt &)> &filter)
        : filter_(filter) {}
    const std::vector<Stmt> &results() const { return results_; }

  protected:
    void visitStmt(const Stmt &op) override;
};

inline std::vector<Stmt>
findStmt(const Stmt &ast, const std::function<bool(const Stmt &)> &filter) {
    FindStmtByFilter visitor(filter);
    visitor(ast);
    return visitor.results();
}

} // namespace ir

#endif // FIND_STMT_H
