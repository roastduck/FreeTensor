#include <autograd/user_grad.h>
#include <visitor.h>

namespace freetensor {

namespace {

class FindDFSFirst : public Visitor {
    const std::unordered_set<ID> &stmts_;
    std::optional<ID> first_;

  public:
    FindDFSFirst(const std::unordered_set<ID> &stmts) : stmts_(stmts) {}

    const auto &first() const { return first_; }

  protected:
    void visitStmt(const Stmt &s) {
        if (!first_.has_value()) {
            if (stmts_.count(s->id())) {
                first_ = s->id();
                return;
            } else {
                Visitor::visitStmt(s);
            }
        } else {
            return;
        }
    }
};

class FindDFSLast : public Visitor {
    const std::unordered_set<ID> &stmts_;
    std::optional<ID> last_;

  public:
    FindDFSLast(const std::unordered_set<ID> &stmts) : stmts_(stmts) {}

    const auto &last() const { return last_; }

  protected:
    void visitStmt(const Stmt &s) {
        Visitor::visitStmt(s);
        if (stmts_.count(s->id())) {
            last_ = s->id();
        }
    }
};

} // Anonymous namespace

std::optional<std::pair<ID, ID>>
getRangeFromStmtSeq(const Stmt &op, const std::unordered_set<ID> &stmts) {
    FindDFSFirst findFirst(stmts);
    FindDFSLast findLast(stmts);
    findFirst(op);
    findLast(op);
    auto &&first = findFirst.first();
    auto &&last = findLast.last();
    if (first.has_value() && last.has_value()) {
        return std::make_pair(*first, *last);
    } else if (!first.has_value() && !last.has_value()) {
        return std::nullopt;
    } else {
        ASSERT(false);
    }
}

} // namespace freetensor
