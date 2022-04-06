#include <itertools.hpp>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <pass/sink_var.h>

namespace ir {

Stmt SinkVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (op->buffer_->atype() != AccessType::Cache || op->pinned_) {
        return op;
    }

    if (!allUses(op->body_).count(op->name_)) {
        return op->body_;
    }

    Stmt ret = op;

    std::vector<VarDef> inners; // outer to inner
    while (op->body_->nodeType() == ASTNodeType::VarDef) {
        auto def = op->body_.as<VarDefNode>();
        for (auto &&dim : def->buffer_->tensor().shape()) {
            if (allReads(dim).count(op->name_)) {
                return ret;
            }
        }
        inners.emplace_back(def);
        op = def;
    }

    switch (op->body_->nodeType()) {
    case ASTNodeType::StmtSeq: {
        auto seq = op->body_.as<StmtSeqNode>();
        int firstUse = -1, lastUse = -1;
        for (auto &&[i, stmt] : iter::enumerate(seq->stmts_)) {
            if (allUses(stmt).count(_op->name_)) {
                if (firstUse == -1) {
                    firstUse = i;
                }
                lastUse = i;
            }
        }
        ASSERT(firstUse != -1);
        if (firstUse > 0 || lastUse < (int)seq->stmts_.size() - 1) {
            Stmt segment;
            if (firstUse == lastUse) {
                segment = seq->stmts_[firstUse];
            } else {
                segment = makeStmtSeq(
                    "", std::vector<Stmt>(seq->stmts_.begin() + firstUse,
                                          seq->stmts_.begin() + lastUse + 1));
            }
            std::vector<Stmt> stmts;
            stmts.reserve(seq->stmts_.size() - (lastUse - firstUse));
            stmts.insert(stmts.end(), seq->stmts_.begin(),
                         seq->stmts_.begin() + firstUse);
            stmts.insert(stmts.end(),
                         makeVarDef(_op->id(), _op->name_, _op->buffer_,
                                    _op->sizeLim_, std::move(segment), false));
            stmts.insert(stmts.end(), seq->stmts_.begin() + lastUse + 1,
                         seq->stmts_.end());
            isFixPoint_ = false;
            ret = makeStmtSeq(seq->id(), std::move(stmts));
            for (auto &&def : iter::reversed(inners)) {
                ret = makeVarDef(def->id(), def->name_, def->buffer_,
                                 def->sizeLim_, std::move(ret), def->pinned_);
            }
        }
        break;
    }

    case ASTNodeType::For: {
        auto loop = op->body_.as<ForNode>();
        // Criteria:
        // 1. All accesses to a variable is indenpendent between each other
        // OR
        // 2. All writes to this variable write the same value
        if (!deps_.count(std::make_pair(_op->name_, loop->id())) ||
            !isVariant(variantMap_, _op, loop->id())) {
            auto loopBody = makeVarDef(_op->id(), _op->name_, _op->buffer_,
                                       _op->sizeLim_, loop->body_, false);
            isFixPoint_ = false;
            ret = makeFor(loop->id(), loop->iter_, loop->begin_, loop->end_,
                          loop->step_, loop->len_, loop->property_,
                          std::move(loopBody));
            for (auto &&def : iter::reversed(inners)) {
                ret = makeVarDef(def->id(), def->name_, def->buffer_,
                                 def->sizeLim_, std::move(ret), def->pinned_);
            }
        }
        break;
    }

    case ASTNodeType::If: {
        auto branch = op->body_.as<IfNode>();
        Stmt thenCase, elseCase;
        thenCase =
            makeVarDef(_op->id().strId() + ".0", _op->name_, _op->buffer_,
                       _op->sizeLim_, branch->thenCase_, false);
        if (branch->elseCase_.isValid()) {
            elseCase =
                makeVarDef(_op->id().strId() + ".1", _op->name_, _op->buffer_,
                           _op->sizeLim_, branch->elseCase_, false);
        }
        ret = makeIf(branch->id(), branch->cond_, std::move(thenCase),
                     std::move(elseCase));
        for (auto &&def : iter::reversed(inners)) {
            ret = makeVarDef(def->id(), def->name_, def->buffer_, def->sizeLim_,
                             std::move(ret), def->pinned_);
        }
        break;
    }

    case ASTNodeType::Assert: {
        auto ass = op->body_.as<AssertNode>();
        auto body = makeVarDef(_op->id(), _op->name_, _op->buffer_,
                               _op->sizeLim_, ass->body_, false);
        ret = makeAssert(ass->id(), ass->cond_, std::move(body));
        for (auto &&def : iter::reversed(inners)) {
            ret = makeVarDef(def->id(), def->name_, def->buffer_, def->sizeLim_,
                             std::move(ret), def->pinned_);
        }
        break;
    }

    default:; // do nothing
    }

    return ret;
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
