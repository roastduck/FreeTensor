#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <container_utils.h>
#include <pass/sink_var.h>

namespace freetensor {

bool SinkVar::hasDep(const ID &vardef, const ID &loop) {
    if (analyzedDeps_.count(vardef)) {
        return deps_.count({vardef, loop});
    } else {
        // Return true (no sink) for now, set `isFixedPoint_ = false` to re-run
        needDepAnalysis_.insert(vardef);
        isFixPoint_ = false;
        return true;
    }
}

Stmt SinkVar::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (toSink_.has_value() && !toSink_->count(op->id())) {
        return op;
    }

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
        for (auto &&dim : def->buffer_->tensor()->shape()) {
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
        for (auto &&[i, stmt] : views::enumerate(seq->stmts_)) {
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
                    std::vector<Stmt>(seq->stmts_.begin() + firstUse,
                                      seq->stmts_.begin() + lastUse + 1));
            }
            std::vector<Stmt> stmts;
            stmts.reserve(seq->stmts_.size() - (lastUse - firstUse));
            stmts.insert(stmts.end(), seq->stmts_.begin(),
                         seq->stmts_.begin() + firstUse);
            stmts.insert(stmts.end(),
                         makeVarDef(_op->name_, _op->buffer_, _op->viewOf_,
                                    std::move(segment), false, _op->metadata(),
                                    _op->id()));
            stmts.insert(stmts.end(), seq->stmts_.begin() + lastUse + 1,
                         seq->stmts_.end());
            isFixPoint_ = false;
            ret = makeStmtSeq(std::move(stmts), seq->metadata(), seq->id());
            for (auto &&def : views::reverse(inners)) {
                ret = makeVarDef(def->name_, def->buffer_, def->viewOf_,
                                 std::move(ret), def->pinned_, def->metadata(),
                                 def->id());
            }
        }
        break;
    }

    case ASTNodeType::For: {
        auto loop = op->body_.as<ForNode>();
        // Criteria:
        // 1. All writes to this variable write the same value
        // OR
        // 2. All accesses to a variable is indenpendent between each other
        if (!isVariant(*variantMap_, _op, loop->id()) ||
            !hasDep(_op->id(), loop->id())) {
            auto loopBody =
                makeVarDef(_op->name_, _op->buffer_, _op->viewOf_, loop->body_,
                           false, _op->metadata(), _op->id());
            isFixPoint_ = false;
            ret = makeFor(loop->iter_, loop->begin_, loop->end_, loop->step_,
                          loop->len_, loop->property_, std::move(loopBody),
                          loop->metadata(), loop->id());
            for (auto &&def : views::reverse(inners)) {
                ret = makeVarDef(def->name_, def->buffer_, def->viewOf_,
                                 std::move(ret), def->pinned_, def->metadata(),
                                 def->id());
            }
        }
        break;
    }

    case ASTNodeType::If: {
        auto branch = op->body_.as<IfNode>();
        if (allReads(branch->cond_).count(_op->name_)) {
            // This check is required or we will crash later passes when
            // encoutering invalid programs like this
            throw InvalidProgram(_op->name_ + " is used before defining");
        }
        Stmt thenCase, elseCase;
        thenCase =
            makeVarDef(_op->name_, _op->buffer_, _op->viewOf_,
                       branch->thenCase_, false, makeMetadata("sink.1", _op));
        if (branch->elseCase_.isValid()) {
            elseCase = makeVarDef(_op->name_, _op->buffer_, _op->viewOf_,
                                  branch->elseCase_, false,
                                  makeMetadata("sink.0", _op));
        }
        ret = makeIf(branch->cond_, std::move(thenCase), std::move(elseCase),
                     branch->metadata(), branch->id());
        for (auto &&def : views::reverse(inners)) {
            ret = makeVarDef(def->name_, def->buffer_, def->viewOf_,
                             std::move(ret), def->pinned_, def->metadata(),
                             def->id());
        }
        break;
    }

    case ASTNodeType::Assert: {
        auto ass = op->body_.as<AssertNode>();
        if (allReads(ass->cond_).count(_op->name_)) {
            // This check is required or we will crash later passes when
            // encoutering invalid programs like this
            throw InvalidProgram(_op->name_ + " is used before defining");
        }
        auto body = makeVarDef(_op->name_, _op->buffer_, _op->viewOf_,
                               ass->body_, false, _op->metadata(), _op->id());
        ret =
            makeAssert(ass->cond_, std::move(body), ass->metadata(), ass->id());
        for (auto &&def : views::reverse(inners)) {
            ret = makeVarDef(def->name_, def->buffer_, def->viewOf_,
                             std::move(ret), def->pinned_, def->metadata(),
                             def->id());
        }
        break;
    }

    default:; // do nothing
    }

    return ret;
}

Stmt sinkVar(const Stmt &_op,
             const std::optional<std::unordered_set<ID>> &toSink) {
    auto op = _op;

    auto variantMap = Lazy([op]() { return findLoopVariance(op).second; });

    auto allLoops = findAllLoops(op);
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop, DepDirection::Normal}});
    }

    std::unordered_set<ID> needDepAnalysis, analyzedDeps;
    std::unordered_set<std::pair<ID, ID>> deps; // {(vardef, loop)}
    for (int i = 0;; i++) {
        if (i > 100) {
            WARNING("SinkVar iterates over 100 rounds. Maybe there is a bug");
            break;
        }

        if (!needDepAnalysis.empty()) {
            FindDeps()
                .direction(direction)
                .filterAccess([&](const AccessPoint &acc) {
                    return needDepAnalysis.count(acc.def_->id());
                })
                .ignoreReductionWAW(false)(op, [&](const Dependency &d) {
                    ASSERT(d.dir_.size() == 1);
                    deps.emplace(d.defId(), d.dir_[0].first.id_);
                });
            for (auto &&vardef : needDepAnalysis) {
                analyzedDeps.insert(vardef);
            }
            needDepAnalysis.clear();
        }

        SinkVar mutator(toSink, deps, analyzedDeps, needDepAnalysis,
                        variantMap);
        op = mutator(op);
        if (mutator.isFixPoint()) {
            break;
        }
    }
    return op;
}

} // namespace freetensor
