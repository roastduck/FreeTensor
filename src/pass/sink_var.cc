#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <analyze/find_all_loops.h>
#include <container_utils.h>
#include <pass/sink_var.h>

namespace freetensor {

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
                         makeVarDef(_op->name_, _op->buffer_, _op->ioTensor_,
                                    std::move(segment), false, _op->metadata(),
                                    _op->id()));
            stmts.insert(stmts.end(), seq->stmts_.begin() + lastUse + 1,
                         seq->stmts_.end());
            isFixPoint_ = false;
            ret = makeStmtSeq(std::move(stmts), seq->metadata(), seq->id());
            for (auto &&def : views::reverse(inners)) {
                ret = makeVarDef(def->name_, def->buffer_, def->ioTensor_,
                                 std::move(ret), def->pinned_, def->metadata(),
                                 def->id());
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
            auto loopBody =
                makeVarDef(_op->name_, _op->buffer_, _op->ioTensor_,
                           loop->body_, false, _op->metadata(), _op->id());
            isFixPoint_ = false;
            ret = makeFor(loop->iter_, loop->begin_, loop->end_, loop->step_,
                          loop->len_, loop->property_, std::move(loopBody),
                          loop->metadata(), loop->id());
            for (auto &&def : views::reverse(inners)) {
                ret = makeVarDef(def->name_, def->buffer_, def->ioTensor_,
                                 std::move(ret), def->pinned_, def->metadata(),
                                 def->id());
            }
        }
        break;
    }

    case ASTNodeType::If: {
        auto branch = op->body_.as<IfNode>();
        Stmt thenCase, elseCase;
        thenCase =
            makeVarDef(_op->name_, _op->buffer_, _op->ioTensor_,
                       branch->thenCase_, false, makeMetadata("sink.1", _op));
        if (branch->elseCase_.isValid()) {
            elseCase = makeVarDef(_op->name_, _op->buffer_, _op->ioTensor_,
                                  branch->elseCase_, false,
                                  makeMetadata("sink.0", _op));
        }
        ret = makeIf(branch->cond_, std::move(thenCase), std::move(elseCase),
                     branch->metadata(), branch->id());
        for (auto &&def : views::reverse(inners)) {
            ret = makeVarDef(def->name_, def->buffer_, def->ioTensor_,
                             std::move(ret), def->pinned_, def->metadata(),
                             def->id());
        }
        break;
    }

    case ASTNodeType::Assert: {
        auto ass = op->body_.as<AssertNode>();
        auto body = makeVarDef(_op->name_, _op->buffer_, _op->ioTensor_,
                               ass->body_, false, _op->metadata(), _op->id());
        ret =
            makeAssert(ass->cond_, std::move(body), ass->metadata(), ass->id());
        for (auto &&def : views::reverse(inners)) {
            ret = makeVarDef(def->name_, def->buffer_, def->ioTensor_,
                             std::move(ret), def->pinned_, def->metadata(),
                             def->id());
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
    std::vector<FindDepsDir> direction;
    direction.reserve(allLoops.size());
    for (auto &&loop : allLoops) {
        direction.push_back({{loop, DepDirection::Normal}});
    }
    std::unordered_set<std::pair<std::string, ID>> deps; // {(var, loop)}
    auto found = [&](const Dependency &d) {
        ASSERT(d.dir_.size() == 1);
        deps.emplace(d.var_, d.dir_[0].first.id_);
    };
    FindDeps().direction(direction)(op, found);

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

} // namespace freetensor
