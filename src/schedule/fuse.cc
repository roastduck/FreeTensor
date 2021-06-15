#include <except.h>
#include <schedule/fuse.h>

namespace ir {

namespace {

struct LoopInVarDefs {
    For loop_;
    std::vector<VarDef> defs_; // inner to outer
};

LoopInVarDefs findLoopInVarDefs(const Stmt &stmt, const std::string &id) {
    if (stmt->id() == id) {
        if (stmt->nodeType() != ASTNodeType::For) {
            throw InvalidSchedule("Statement " + id + " is not a loop");
        }
        return LoopInVarDefs{stmt.as<ForNode>(), {}};
    }
    if (stmt->nodeType() == ASTNodeType::VarDef) {
        auto ret = findLoopInVarDefs(stmt.as<VarDefNode>()->body_, id);
        ret.defs_.emplace_back(stmt.as<VarDefNode>());
        return ret;
    }
    return LoopInVarDefs{nullptr, {}};
}

} // Anonymous namespace

Expr FuseFor::visit(const Var &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Var);
    auto op = __op.as<VarNode>();
    if (op->name_ == iter0_) {
        return makeAdd(makeVar(iter0_), begin0_);
    }
    if (op->name_ == iter1_) {
        // Yes, use iter0_
        return makeAdd(makeVar(iter0_), begin1_);
    }
    return op;
}

Stmt FuseFor::visit(const For &_op) {
    if (_op->id() == id0_) {
        iter0_ = _op->iter_, begin0_ = _op->begin_;
    }
    if (_op->id() == id1_) {
        iter1_ = _op->iter_, begin1_ = _op->begin_;
    }
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    if (op->id() == id0_ || op->id() == id1_) {
        return makeFor(op->id(), op->iter_, makeIntConst(0), op->len_, op->len_,
                       op->parallel_, op->unroll_, op->vectorize_, op->body_);
    }
    return op;
}

Stmt FuseFor::visit(const StmtSeq &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        auto loop0InVarDefs = findLoopInVarDefs(op->stmts_[i], id0_);
        if (loop0InVarDefs.loop_.isValid()) {
            if (i + 1 == iEnd) {
                throw InvalidSchedule("Fuse: Loop " + id0_ + " and " + id1_ +
                                      " shuold be directly following");
            }
            auto loop1InVarDefs = findLoopInVarDefs(op->stmts_[i + 1], id1_);
            if (!loop1InVarDefs.loop_.isValid()) {
                throw InvalidSchedule("Fuse: Loop " + id0_ + " and " + id1_ +
                                      " shuold be directly following");
            }

            auto loop0 = loop0InVarDefs.loop_;
            auto loop1 = loop1InVarDefs.loop_;
            auto fused = makeFor(fused_, iter0_, makeIntConst(0), loop0->end_,
                                 loop0->end_, loop0->parallel_, loop0->unroll_,
                                 loop0->vectorize_,
                                 makeStmtSeq("", {loop0->body_, loop1->body_}));

            // From inner to outer
            // FIXME: Check the VarDefs can really be hoisted via
            // check_not_modified
            for (auto &&def : loop1InVarDefs.defs_) {
                fused =
                    makeVarDef(def->id(), def->name_, std::move(*def->buffer_),
                               def->sizeLim_, fused, def->pinned_);
            }
            for (auto &&def : loop0InVarDefs.defs_) {
                fused =
                    makeVarDef(def->id(), def->name_, std::move(*def->buffer_),
                               def->sizeLim_, fused, def->pinned_);
            }

            op->stmts_[i] =
                makeAssert("", makeEQ(loop0->end_, loop1->end_), fused);
            op->stmts_.erase(op->stmts_.begin() + i + 1);
            break;
        }
    }
    return op;
}

} // namespace ir
