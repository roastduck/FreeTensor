#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
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

Stmt SinkVar::visit(const ReduceTo &op) {
    used_.insert(op->var_);
    return Mutator::visit(op);
}

Stmt SinkVar::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache || op->pinned_) {
        return Mutator::visit(op);
    }

    std::vector<Expr> shape;
    shape.reserve(op->buffer_->tensor().shape().size());
    for (auto &&dim : op->buffer_->tensor().shape()) {
        shape.emplace_back((*this)(dim));
    }
    Tensor tensor(std::move(shape), op->buffer_->tensor().dtype());
    Buffer buffer(std::move(tensor), op->buffer_->atype(),
                  op->buffer_->mtype());
    Expr sizeLim = op->sizeLim_.isValid() ? (*this)(op->sizeLim_) : nullptr;
    Stmt body;

    switch (op->body_->nodeType()) {
    case ASTNodeType::StmtSeq: {
        auto seq = op->body_.as<StmtSeqNode>();
        int firstUse = -1, lastUse = -1;
        std::vector<Stmt> stmts;
        stmts.reserve(seq->stmts_.size());
        for (size_t i = 0, iEnd = seq->stmts_.size(); i < iEnd; i++) {
            used_.erase(op->name_);
            stmts.emplace_back((*this)(seq->stmts_[i]));
            if (used_.count(op->name_)) {
                if (firstUse == -1) {
                    firstUse = i;
                }
                lastUse = i;
            }
        }
        if (firstUse == -1) {
            isFixPoint_ = false;
            return makeStmtSeq(seq->id(), std::move(stmts));
        } else if (firstUse > 0 || lastUse < (int)seq->stmts_.size() - 1) {
            Stmt segment;
            if (firstUse == lastUse) {
                segment = stmts[firstUse];
            } else {
                segment = makeStmtSeq(
                    "", std::vector<Stmt>(stmts.begin() + firstUse,
                                          stmts.begin() + lastUse + 1));
            }
            stmts.erase(stmts.begin() + firstUse, stmts.begin() + lastUse + 1);
            stmts.insert(stmts.begin() + firstUse,
                         makeVarDef(op->id(), op->name_, std::move(buffer),
                                    std::move(sizeLim), std::move(segment),
                                    false));
            isFixPoint_ = false;
            return makeStmtSeq(seq->id(), std::move(stmts));
        } else {
            body = makeStmtSeq(seq->id(), std::move(stmts));
        }
        break;
    }

    case ASTNodeType::For: {
        auto loop = op->body_.as<ForNode>();
        // Criteria:
        // 1. All accesses to a variable is indenpendent between each other
        // OR
        // 2. All writes to this variable write the same value
        if (!deps_.count(std::make_pair(op->name_, loop->id())) ||
            !isVariant(variantMap_, op, loop->id())) {
            auto loopBody =
                makeVarDef(op->id(), op->name_, std::move(buffer),
                           std::move(sizeLim), (*this)(loop->body_), false);
            isFixPoint_ = false;
            return makeFor(loop->id(), loop->iter_, (*this)(loop->begin_),
                           (*this)(loop->end_), (*this)(loop->step_),
                           (*this)(loop->len_), loop->property_,
                           std::move(loopBody));
        } else {
            body = (*this)(op->body_);
        }
        break;
    }

    default:
        body = (*this)(op->body_);
    }

    return makeVarDef(op->id(), op->name_, std::move(buffer),
                      std::move(sizeLim), body, false);
}

Stmt sinkVar(const Stmt &_op) {
    auto op = _op;

    auto allLoops = findAllLoops(op);
    std::vector<FindDepsCond> cond;
    cond.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        cond.push_back({{loop, DepDirection::Normal}});
    }
    std::unordered_set<std::pair<std::string, ID>> deps; // {(var, loop)}
    auto found = [&](const Dependency &d) {
        ASSERT(d.cond_.size() == 1);
        deps.emplace(d.var_, d.cond_[0].first.id_);
    };
    findDeps(op, cond, found);

    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING("SinkVar iterates over 100 rounds. Maybe there is a bug");
            break;
        }

        // findLoopVariance returns AST node reference instead of IDs, so we
        // need it once per mutation
        LoopVariUniqVarMap variantMap = findLoopVariance(op).second;

        SinkVar mutator(deps, variantMap);
        op = mutator(op);
        if (mutator.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace ir
