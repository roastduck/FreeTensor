#include <pass/sink_var.h>

namespace ir {

Expr SinkVar::visit(const Load &op) {
    used_.insert(op->var_);
    return Mutator::visit(op);
}

Stmt SinkVar::visit(const Store &op) {
    used_.insert(op->var_);
    return Mutator::visit(op);
}

Stmt SinkVar::visit(const AddTo &op) {
    used_.insert(op->var_);
    return Mutator::visit(op);
}

Stmt SinkVar::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache) {
        return Mutator::visit(op);
    }

    std::vector<Expr> shape;
    shape.reserve(op->buffer_->tensor().shape().size());
    for (auto &&dim : op->buffer_->tensor().shape()) {
        shape.emplace_back((*this)(dim));
    }
    Tensor tensor(std::move(shape), op->buffer_->tensor().dtype());
    Buffer buffer(std::move(tensor), op->buffer_->atype());
    Stmt body;

    switch (op->body_->nodeType()) {
    case ASTNodeType::StmtSeq: {
        auto seq = op->body_.as<StmtSeqNode>();
        size_t lastUse = 0, useCnt = 0;
        std::vector<Stmt> stmts;
        stmts.reserve(seq->stmts_.size());
        for (size_t i = 0, iEnd = seq->stmts_.size(); i < iEnd; i++) {
            used_.erase(op->name_);
            stmts.emplace_back((*this)(seq->stmts_[i]));
            if (used_.count(op->name_)) {
                lastUse = i, useCnt++;
            }
        }
        if (useCnt == 0) {
            isFixPoint_ = false;
            return makeStmtSeq(seq->id(), std::move(stmts));
        } else if (useCnt == 1) {
            stmts[lastUse] = makeVarDef(op->id(), op->name_, std::move(buffer),
                                        stmts[lastUse]);
            isFixPoint_ = false;
            return makeStmtSeq(seq->id(), std::move(stmts));
        } else {
            body = makeStmtSeq(seq->id(), std::move(stmts));
        }
        break;
    }

    default:
        body = (*this)(op->body_);
    }

    return makeVarDef(op->id(), op->name_, std::move(buffer), body);
}

Stmt sinkVar(const Stmt &_op) {
    Stmt op = _op;
    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING("SinkVar iterates over 100 rounds. Maybe there is a bug");
            break;
        }
        SinkVar mutator;
        op = mutator(op);
        if (mutator.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace ir

