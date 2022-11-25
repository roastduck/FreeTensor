#include <debug/check_conflict_id.h>
#include <except.h>
#include <visitor.h>

namespace freetensor {

namespace {

class CheckConflictId : public Visitor {
    std::unordered_map<ID, Stmt> firstOccur_;

  protected:
    void visitStmt(const Stmt &op) override {
        if (firstOccur_.contains(op->id())) {
            std::ostringstream oss;
            oss << "ID " << op->id() << " of " << op
                << " conflicts with its first occurrence at "
                << firstOccur_.at(op->id());
            throw InvalidProgram(oss.str());
        } else {
            firstOccur_[op->id()] = op;
            Visitor::visitStmt(op);
        }
    }
};

} // namespace

void checkConflictId(const Stmt &ast) { CheckConflictId()(ast); }

} // namespace freetensor
