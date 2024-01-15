#include <unordered_map>

#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <container_utils.h>
#include <get_new_name.h>
#include <math/min_max.h>
#include <math/parse_pb_expr.h>
#include <math/presburger.h>
#include <mutator.h>
#include <pass/replace_iter.h>
#include <pass/shrink_for.h>
#include <schedule.h>
#include <schedule/parallelize.h>
#include <schedule/parallelize_as.h>

namespace freetensor {

namespace {

PBMap projectOntoOneOutputDim(const PBMap &map, int dim) {
    auto ret = map;
    if (dim < (int)map.nOutDims() - 1) {
        ret = projectOutOutputDims(std::move(ret), dim + 1,
                                   map.nOutDims() - dim - 1);
    }
    if (dim > 0) {
        ret = projectOutOutputDims(std::move(ret), 0, dim);
    }
    ASSERT(ret.nOutDims() == 1);
    return ret;
}

class AddParScopes : public TrackStmt<SymbolTable<Mutator>> {
    typedef TrackStmt<SymbolTable<Mutator>> BaseClass;

    ID nest_;
    const PBCtx &presburger_;
    const std::vector<For> &orderedScopes_;
    const std::unordered_map<ID, PBMap> &scope2Idx2Iter_;

    ID newNestId_;
    std::vector<std::string> newIterNames_;
    std::vector<ID> newScopeIds_;

    bool inside_ = false;

    // Itmes of inner vectors are from scopes, which are combined by LAnd. Items
    // of the outer vector are from access sites, which are combined by LOr, so
    // they can be checked by subsequent `parallelize` schedules.
    std::unordered_map<ID, std::vector<std::vector<Expr>>> threadGuard_;

  public:
    AddParScopes(const ID &nest, const PBCtx &presburger,
                 const std::vector<For> &orderedScopes,
                 const std::unordered_map<ID, PBMap> &scope2Idx2Iter)
        : nest_(nest), presburger_(presburger), orderedScopes_(orderedScopes),
          scope2Idx2Iter_(scope2Idx2Iter) {}

    const auto &newScopeIds() const { return newScopeIds_; }
    const auto &newNestId() const { return newNestId_; }

  private:
    template <typename T> auto visitAcc(const T &op) {
        if (inside_) {
            std::vector<Expr> thisThreadGuard;
            thisThreadGuard.reserve(orderedScopes_.size());
            for (auto &&[scope, newIterName] :
                 views::zip(orderedScopes_, newIterNames_)) {
                auto &&idx2iter = coalesce(scope2Idx2Iter_.at(scope->id()));
                SimplePBFuncAST f;
                try {
                    f = parseSimplePBFuncReconstructMinMax(presburger_,
                                                           idx2iter);
                } catch (const ParserError &e) {
                    throw InvalidSchedule(
                        FT_MSG << "Thread mapping is not a simple function: "
                               << e.what());
                }
                ASSERT(f.args_.size() == op->indices_.size());
                ASSERT(f.values_.size() == 1);
                std::unordered_map<std::string, Expr> replace;
                replace.reserve(op->indices_.size());
                for (auto &&[arg, idx] : views::zip(f.args_, op->indices_)) {
                    replace[arg] = idx;
                }
                thisThreadGuard.emplace_back(
                    makeEQ(makeVar(newIterName),
                           ReplaceIter{replace}(f.values_.front())));
            }
            for (Stmt s = curStmt(); s.isValid(); s = s->parentStmt()) {
                threadGuard_[s->id()].emplace_back(std::move(thisThreadGuard));
                if (s->id() == nest_) {
                    break;
                }
            }
        }
        return BaseClass::visit(op);
    }

    Stmt doVisitStmt(const Stmt &s) {
        auto ret = BaseClass::visitStmt(s);
        if (inside_) {
            if (auto it = threadGuard_.find(s->id());
                it != threadGuard_.end()) {
                ret = makeIf(makeLOrLAnd(it->second), std::move(ret));
            }
        }
        return ret;
    }

  protected:
    using BaseClass::visit;

    Stmt visitStmt(const Stmt &s) override {
        if (s->id() == nest_) {
            auto usedNames = uni(names(), allNames(s));
            for (auto &&scope : views::reverse(orderedScopes_)) {
                auto newIterName = getNewName(scope->iter_, usedNames);
                usedNames.emplace(newIterName);
                newIterNames_.emplace_back(newIterName);
            }

            ASSERT(!inside_);
            inside_ = true;
            auto ret = doVisitStmt(s);
            inside_ = false;

            for (auto &&[scope, newIterName] :
                 views::reverse(views::zip(orderedScopes_, newIterNames_))) {
                // Make `For`s with empty parallel_ property, and subsequent
                // `parallelize` schedules will parallelize them and check the
                // legality.
                ret = makeFor(newIterName, scope->begin_, scope->end_,
                              scope->step_, scope->len_,
                              Ref<ForProperty>::make(), std::move(ret));
                newScopeIds_.emplace(newScopeIds_.begin(), ret->id());
            }

            newNestId_ = ret->id();
            return ret;
        } else {
            return doVisitStmt(s);
        }
    }

    Expr visit(const Load &op) override { return visitAcc(op); }
    Stmt visit(const Store &op) override { return visitAcc(op); }
    Stmt visit(const ReduceTo &op) override { return visitAcc(op); }
};

} // Anonymous namespace

Stmt parallelizeAs(const Stmt &_ast, const ID &nest, const ID &reference,
                   const ID &defId) {
    Stmt ast = _ast;

    bool referenceIsBeforeNest =
        findStmt(ast, reference)->isBefore(findStmt(ast, nest));
    auto isSafeToMove = [&](const Expr &expr) {
        // To be conservative, check all the range enclosing reference and nest
        if (referenceIsBeforeNest) {
            return checkNotModified(ast, expr, CheckNotModifiedSide::Before,
                                    reference, CheckNotModifiedSide::After,
                                    nest);
        } else {
            return checkNotModified(ast, expr, CheckNotModifiedSide::Before,
                                    nest, CheckNotModifiedSide::After,
                                    reference);
        }
    };
    auto checkSafeToMoveOrThrow = [&](const Expr &expr) {
        if (!isSafeToMove(expr)) {
            throw InvalidSchedule(
                FT_MSG << expr
                       << " in a reference nest's loop range is not supported");
        }
    };

    FindAccessPoint finder{
        defId, DEP_ALL, syncFunc([&](const Access &acc) {
            return acc.stmt_->ancestorById(reference).isValid();
        })};
    finder.doFind(ast);

    PBCtx presburger;
    std::unordered_map<ID, PBMap> scope2Idx2Iter;
    for (const Ref<AccessPoint> &acc :
         views::concat(finder.reads(), finder.writes())) {
        GenPBExpr::VarMap externals;
        auto iter2idx = AnalyzeDeps::makeAccMapStatic(
            presburger, *acc, acc->iter_.size(), acc->access_.size(),
            RelaxMode::Possible, "", externals, {}, true);
        if (!externals.empty()) {
            throw InvalidSchedule(
                FT_MSG << "Indirect thread mapping in reference loop nest "
                       << reference << " is not supported");
        }

        std::unordered_map<std::string, For> iter2Scope;
        for (Stmt s = acc->stmt_; s.isValid(); s = s->parentStmt()) {
            if (s->nodeType() == ASTNodeType::For) {
                if (auto &&loop = s.as<ForNode>();
                    loop->property_->parallel_ != serialScope) {
                    iter2Scope[loop->iter_] = loop;
                }
            }
            if (s->id() == reference) {
                break;
            }
        }

        for (auto &&[i, iterAxis] : views::enumerate(acc->iter_)) {
            if (iterAxis.iter_->nodeType() == ASTNodeType::Var) {
                if (auto it =
                        iter2Scope.find(iterAxis.iter_.as<VarNode>()->name_);
                    it != iter2Scope.end()) {
                    auto &&id = it->second->id();
                    auto thisIdx2Iter =
                        projectOntoOneOutputDim(reverse(iter2idx), i);
                    scope2Idx2Iter[id] =
                        scope2Idx2Iter.count(id)
                            ? uni(scope2Idx2Iter[id], thisIdx2Iter)
                            : thisIdx2Iter;
                }
            }
        }
    }

    std::vector<For> orderedScopes;
    for (auto &&s : findAllStmt(ast, "(<For><<-" + toString(reference) + ")|" +
                                         toString(reference))) {
        ASSERT(s->nodeType() == ASTNodeType::For);
        auto &&loop = s.as<ForNode>();
        checkSafeToMoveOrThrow(loop->begin_);
        checkSafeToMoveOrThrow(loop->end_);
        checkSafeToMoveOrThrow(loop->step_);
        if (std::ranges::find_if(orderedScopes, [&](const For &f) {
                return f->property_->parallel_ == loop->property_->parallel_;
            }) != orderedScopes.end()) {
            throw InvalidSchedule(
                FT_MSG << "Multiple loops bound to the same parallel scope "
                       << loop->property_->parallel_
                       << " in the reference loop nest " << reference
                       << " is not supported yet");
        }
        if (auto it = scope2Idx2Iter.find(loop->id());
            it != scope2Idx2Iter.end()) {
            if (!it->second.isSingleValued()) {
                throw InvalidSchedule(
                    FT_MSG << "Reference loop nest " << reference
                           << " is not thread-local w.r.t scope " << loop->id()
                           << ". The mapping from indices to iterator is "
                           << it->second << ".");
            }
            orderedScopes.emplace_back(s.as<ForNode>());
        }
    }

    AddParScopes adder{nest, presburger, orderedScopes, scope2Idx2Iter};
    ast = adder(ast);

    // Shrink original loops in `nest` according to the gaurds with just add. If
    // the loop does not carry dependences, we can use a more aggressive
    // "unordered" shrinking.
    std::vector<FindDepsDir> dirs;
    for (auto &&s : findAllStmt(ast, "(<For><<-" + toString(nest) + ")|" +
                                         toString(nest))) {
        dirs.push_back({{s->id(), DepDirection::Normal}});
    }
    bool unordered = !FindDeps().direction(dirs).filterSubAST(nest).exists(ast);
    ast = shrinkFor(ast, nest, true, unordered);

    for (auto &&[id, scope] : views::zip(adder.newScopeIds(), orderedScopes)) {
        ast = parallelize(ast, id, scope->property_->parallel_, true);
    }

    return ast;
}

void Schedule::parallelizeAs(const ID &nest, const ID &reference,
                             const ID &defId) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(
        ParallelizeAs, freetensor::parallelizeAs, nest, reference, defId));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
