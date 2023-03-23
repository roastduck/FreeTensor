#include <sstream>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <autograd/invert_stmts.h>
#include <math/parse_pb_expr.h>
#include <mutator.h>
#include <pass/make_nested_loops.h>
#include <pass/replace_iter.h>

namespace freetensor {

namespace {

class InsertLifetimeEndChecker : public Mutator {
    const std::unordered_map<ID, std::unordered_set<ID>> &idsNeeded_;

  public:
    InsertLifetimeEndChecker(
        const std::unordered_map<ID, std::unordered_set<ID>> &idsNeeded)
        : idsNeeded_(idsNeeded) {}

  protected:
    Stmt visit(const VarDef &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        if (idsNeeded_.count(op->id())) {
            auto n = op->buffer_->tensor()->shape().size();
            std::vector<std::string> fakeIters;
            std::vector<Expr> fakeIndices;
            fakeIters.reserve(n);
            fakeIndices.reserve(n);
            for (size_t i = 0; i < n; i++) {
                auto var = "__lifetime_checker_i" + std::to_string(i);
                fakeIters.emplace_back(var);
                fakeIndices.emplace_back(makeVar(var));
            }
            auto fakeExpr = makeIntrinsic(
                "__any__", {}, op->buffer_->tensor()->dtype(), false);
            auto fakeStore = makeStore(op->name_, std::move(fakeIndices),
                                       std::move(fakeExpr));
            auto fakeLoop = makeNestedLoops(
                fakeIters, views::repeat(makeIntConst(0)),
                op->buffer_->tensor()->shape(), views::repeat(makeIntConst(1)),
                op->buffer_->tensor()->shape(),
                views::repeat(Ref<ForProperty>::make()), std::move(fakeStore));
            op->body_ = makeStmtSeq({op->body_, fakeLoop});
        }
        return op;
    }
};

struct CondInfo {
    std::vector<IterAxis> iter_;
    PBSet when_;
    Expr cond_;
};

PBMap anythingTo1(PBCtx &presburger, int nDims) {
    std::ostringstream os;
    os << "{[";
    for (int i = 0; i < nDims; i++) {
        os << (i > 0 ? "," : "") << "i" << i;
    }
    os << "] -> [1]}";
    return PBMap(presburger, os.str());
}

void genCondExpr(PBCtx &presburger, CondInfo *info) {
    // Use less basic sets to express info->when_. It also makes our final
    // expression simpler
    info->when_ = coalesce(info->when_);

    try {
        // Build a map to express whether an iteration point is in
        // `info->when_`. Note that we are not using
        // `isl_set_indicator_function` here, because we only need `{x
        // -> 1: x is in set}`, and we don't need `{x -> 0: x is not in set}`.
        // The latter may require multiple basic maps to express
        PBMap indicator = intersectDomain(
            anythingTo1(presburger, info->when_.nDims()), info->when_);
        for (auto &&[args, _, factorRange] :
             parsePBFunc(toString(PBFunc(indicator)))) {
            std::unordered_map<std::string, Expr> islVarToIter;
            for (auto &&[iter, arg] : views::zip(info->iter_, args)) {
                islVarToIter[arg] = !iter.negStep_
                                        ? iter.iter_
                                        : makeMul(makeIntConst(-1), iter.iter_);
            }
            ReplaceIter replacer{islVarToIter};
            info->cond_ = info->cond_.isValid()
                              ? makeLOr(info->cond_, replacer(factorRange))
                              : replacer(factorRange);
        }
    } catch (const ParserError &e) {
        // Leave info.cond_ as empty to indicate we have failed
    }
}

class InsertSelfAssign : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, CondInfo> &unrecoverableInfo_;
    std::unordered_map<ID, std::unordered_set<ID>> *idsNeeded_;
    std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        *derivatives_;

  public:
    InsertSelfAssign(
        const std::unordered_map<ID, CondInfo> &unrecoverableInfo,
        std::unordered_map<ID, std::unordered_set<ID>> *idsNeeded,
        std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
            *derivatives)
        : unrecoverableInfo_(unrecoverableInfo), idsNeeded_(idsNeeded),
          derivatives_(derivatives) {}

    template <typename T> Stmt doVisit(const T &_op) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (auto it = unrecoverableInfo_.find(op->id());
            it != unrecoverableInfo_.end()) {
            if (!it->second.cond_.isValid()) {
                return op;
            }
            auto self = makeLoad(op->var_, op->indices_,
                                 buffer(op->var_)->tensor()->dtype());
            auto selfAssign = makeStore(op->var_, op->indices_, self);

            // The self-assignment must be taped or recomputed
            idsNeeded_->at(def(op->var_)->id()).emplace(selfAssign->id());

            // Update derivative info of this self-assignment
            (*derivatives_)[StmtOrExprID{self, selfAssign}].addPartial(
                self.template as<LoadNode>(),
                Derivative::LazyPartialDerivative{
                    symbolTableSnapshot(), makeIntConst(1), selfAssign->id()});

            return makeStmtSeq({op, makeIf(it->second.cond_, selfAssign)});
        }
        return op;
    }

  protected:
    using BaseClass::visit;
    Stmt visit(const Store &op) override { return doVisit(op); }
    Stmt visit(const ReduceTo &op) override { return doVisit(op); }
};

} // Anonymous namespace

void FindInvertibles::visit(const ReduceTo &op) {
    BaseClass::visit(op);

    auto targetDType = buffer(op->var_)->tensor()->dtype();
    auto exprDType = op->expr_->dtype();
    // There should not be implicit rounding. The frontend has ensured this
    ASSERT(targetDType.base() == exprDType.base());

    if (allReads(op->expr_).count(op->var_)) {
        return; // Unsupported
    }
    switch (op->op_) {
    case ReduceOp::Add:
        invertibles_[op->id()] = makeStore(
            op->var_, op->indices_,
            makeSub(makeLoad(op->var_, op->indices_, targetDType), op->expr_));
        break;
    case ReduceOp::Mul:
        if (isNE0(exprDType)) {
            if (isFloat(targetDType)) {
                invertibles_[op->id()] = makeStore(
                    op->var_, op->indices_,
                    makeRealDiv(makeLoad(op->var_, op->indices_, targetDType),
                                op->expr_));
            }
            if (isInt(targetDType)) {
                invertibles_[op->id()] = makeStore(
                    op->var_, op->indices_,
                    makeFloorDiv(makeLoad(op->var_, op->indices_, targetDType),
                                 op->expr_));
            }
        }
        break;
    default:; // Ignore
    }
}

std::tuple<Stmt, std::unordered_map<ID, InversionInfo>>
invertStmts(const Stmt &op,
            std::unordered_map<ID, std::unordered_set<ID>> *idsNeeded,
            std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
                *derivatives) {
    FindInvertibles finder;
    finder(op);
    auto &&invertibles = finder.invertibles();
    if (invertibles.empty()) {
        return {op, {}};
    }

    // If we want to recover a statement X's value fully by inverting other
    // statements, X is recoverable only if in every cases there is a invertible
    // statement overwritting it, and X is not allowed to reach the end of its
    // lifetime without being overwritten.
    //
    // This requirement is too strict, so we allow partially recover a
    // statement: if in some condition `cond` the statement X cannot be
    // recovered, we insert a self-assigning statement Y like:
    //
    // ```
    // a[...] += ... ------------- Statement X
    // if (cond) {
    //   a[...] = a[...] --------- Statement Y
    // }
    // ```
    //
    // Y is to be recomputed in the traditional way, and now we can safely
    // inversely recover X in all cases.
    //
    // In order to get `cond`, we analyze when X is overwritten by a non-
    // invertible statement, and when X reaches the end of its lifetime. To
    // detect the latter case, we insert a fake self-assigning statement just
    // before the lifetime's end before dependence analysis.
    PBCtx presburger;
    std::unordered_map<ID, CondInfo> unrecoverableInfo, toInvertInfo;
    std::unordered_map<ID, PBSet> allPossibleIters;
    // TODO: We can apply an additional filter to only invert Y if it already to
    // be taped or recomputed. This way, inverting Y won't cause any additional
    // values to be recomputed.
    FindDeps()
        .type(DEP_WAW)
        .ignoreReductionWAW(false)
        .noProjectOutPrivateAxis(true) // because we need presburger maps
        .filterEarlier([&](const auto &acc) -> bool {
            if (auto it = idsNeeded->find(acc.def_->id());
                it != idsNeeded->end()) {
                return it->second.count(acc.stmt_->id());
            }
            return false;
        })(InsertLifetimeEndChecker{*idsNeeded}(op), [&](const Dependence &d) {
            // Serialize and deserialize to change PBCtx
            auto earlierIterSet =
                PBSet(presburger, toString(range(d.later2EarlierIter_)));

            auto toInvert = d.later_.stmt_->id();
            auto toRecover = d.earlier_.stmt_->id();
            if (invertibles.count(d.later_.stmt_->id())) { // Can invert
                // Serialize and deserialize to change PBCtx
                auto laterIterSet =
                    PBSet(presburger, toString(domain(d.later2EarlierIter_)));

                if (!toInvertInfo.count(toInvert)) {
                    toInvertInfo[toInvert] =
                        CondInfo{d.later_.iter_, laterIterSet, nullptr};
                } else {
                    toInvertInfo[toInvert].when_ =
                        uni(toInvertInfo.at(toInvert).when_, laterIterSet);
                }
            } else { // Cannot invert
                if (!unrecoverableInfo.count(toRecover)) {
                    unrecoverableInfo[toRecover] =
                        CondInfo{d.earlier_.iter_, earlierIterSet, nullptr};
                } else {
                    unrecoverableInfo[toRecover].when_ = uni(
                        unrecoverableInfo.at(toRecover).when_, earlierIterSet);
                }
            }

            if (!allPossibleIters.count(toRecover)) {
                allPossibleIters[toRecover] = earlierIterSet;
            } else {
                allPossibleIters[toRecover] =
                    uni(allPossibleIters.at(toRecover), earlierIterSet);
            }
        });
    for (auto &&[id, info] : unrecoverableInfo) {
        if (info.when_ == allPossibleIters.at(id)) {
            continue; // Always unrecoverable
        }
        genCondExpr(presburger, &info);
    }
    for (auto &&[id, info] : toInvertInfo) {
        genCondExpr(presburger, &info);
    }
    auto ret = InsertSelfAssign{unrecoverableInfo, idsNeeded, derivatives}(op);

    // Check which statements can be successfully recovered
    std::unordered_map<ID, std::unordered_set<ID>> canRecover;
    for (auto &&[defId, ids] : *idsNeeded) {
        for (auto &&id : ids) {
            if (auto it = unrecoverableInfo.find(id);
                it != unrecoverableInfo.end()) {
                if (!it->second.cond_.isValid()) {
                    continue; // Cannot get cond
                }
            }
            canRecover[defId].emplace(id);
        }
    }
    for (auto &&[defId, ids] : canRecover) {
        for (auto &&id : ids) {
            idsNeeded->at(defId).erase(id);
        }
    }

    // Merge `invertibles` and `toInvertInfo` to output format
    std::unordered_map<ID, InversionInfo> toInvert;
    for (auto &&[id, info] : toInvertInfo) {
        toInvert[id] = InversionInfo{invertibles.at(id), info.cond_};
    }

    return {ret, toInvert};
}

} // namespace freetensor
