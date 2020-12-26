#include <analyze/deps.h>
#include <except.h>

namespace ir {

void FindAccessPoint::visit(const StmtSeq &op) {
    cur_.emplace_back();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        cur_.back() = makeIntConst(i);
        (*this)(op->stmts_[i]);
    }
    cur_.pop_back();
}

void FindAccessPoint::visit(const For &op) {
    cur_.emplace_back(makeVar(op->iter_));
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const Store &op) {
    Visitor::visit(op);
    writes_[op.get()] = cur_;
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    reads_[op.get()] = cur_;
}

AccessPoint AnalyzeDeps::makeDep(const AccessPoint &lhs,
                                 const AccessPoint &rhs) {
    size_t n = std::min(lhs.size(), rhs.size());
    AccessPoint ret;
    ret.reserve(n);
    for (size_t i = 0; i < n; i++) {
        ret.emplace_back(makeSub(lhs[i], rhs[i]));
    }
    return ret;
}

Stmt AnalyzeDeps::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    std::vector<AccessPoint> rw, ww;
    auto &&point = writes_.at(op.get());
    for (auto &&other : reads_) {
        rw.emplace_back(makeDep(point, other.second));
    }
    for (auto &&other : writes_) {
        ww.emplace_back(makeDep(point, other.second));
    }
    op->info_dep_rw_ = std::move(rw);
    op->info_dep_ww_ = std::move(ww);
    return op;
}

Expr AnalyzeDeps::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    std::vector<AccessPoint> rw;
    auto &&point = reads_.at(op.get());
    for (auto &&other : writes_) {
        rw.emplace_back(makeDep(point, other.second));
    }
    op->info_dep_rw_ = std::move(rw);
    return op;
}

Stmt analyzeDeps(const Stmt &op) {
    FindAccessPoint visitor;
    visitor(op);
    AnalyzeDeps mutator(visitor.reads(), visitor.writes());
    return mutator(op);
}

} // namespace ir

