#include <algorithm>
#include <regex>
#include <sstream>

#include <itertools.hpp>

#include <analyze/deps.h>
#include <container_utils.h>
#include <except.h>
#include <mutator.h>
#include <pass/simplify.h>
#include <serialize/mangle.h>

namespace freetensor {

void FindAllNoDeps::visit(const For &op) {
    Visitor::visit(op);
    for (auto &&var : op->property_->noDeps_) {
        results_[var].emplace_back(op->id());
    }
}

void CountBandNodeWidth::visit(const Load &op) {
    // No recursion
    if (!lastIsLoad_) {
        width_++;
        lastIsLoad_ = true;
    }
}

void CountBandNodeWidth::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);
    width_++;
    lastIsLoad_ = false;
}

void CountBandNodeWidth::visit(const Store &op) {
    Visitor::visit(op);
    width_++;
    lastIsLoad_ = false;
}

void CountBandNodeWidth::visit(const ReduceTo &op) {
    Visitor::visit(op);
    width_++;
    lastIsLoad_ = false;
}

FindAccessPoint::FindAccessPoint(const Stmt &root) {
    if (int width = countBandNodeWidth(root); width > 1) {
        cur_.emplace_back(makeIntConst(-1));
    }
}

void FindAccessPoint::visit(const VarDef &op) {
    allDefs_.emplace_back(op);
    defAxis_[op->name_] =
        !cur_.empty() && cur_.back().iter_->nodeType() == ASTNodeType::IntConst
            ? cur_.size() - 1
            : cur_.size();
    BaseClass::visit(op);
    defAxis_.erase(op->name_);
}

void FindAccessPoint::visit(const StmtSeq &op) {
    if (!cur_.empty() &&
        cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
        scope2coord_[op->id()] = cur_;
    }
    BaseClass::visit(op);
}

void FindAccessPoint::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    if (!cur_.empty() &&
        cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
        // top is band node
        cur_.back().iter_ =
            makeIntConst(cur_.back().iter_.as<IntConstNode>()->val_ + 1);
    }
    lastIsLoad_ = false;

    auto iter = makeVar(op->iter_);
    auto oldCondsSize = conds_.size();
    // We use IfExpr instead of determine the sign of op->step_ here, because
    // GenPBExpr can fold the constants
    auto posiCond = makeLAnd(
        makeLAnd(makeGE(iter, op->begin_), makeLT(iter, op->end_)),
        makeEQ(makeMod(makeSub(iter, op->begin_), op->step_), makeIntConst(0)));
    auto negCond = makeLAnd(
        makeLAnd(makeLE(iter, op->begin_), makeGT(iter, op->end_)),
        makeEQ(makeMod(makeSub(iter, op->begin_), op->step_), makeIntConst(0)));
    auto zeroCond = makeEQ(iter, op->begin_);
    conds_.emplace_back(
        makeIfExpr(makeGT(op->step_, makeIntConst(0)), std::move(posiCond),
                   makeIfExpr(makeLT(op->step_, makeIntConst(0)),
                              std::move(negCond), zeroCond)));
    cur_.emplace_back(iter, op->property_->parallel_);
    scope2coord_[op->id()] = cur_;
    if (int width = countBandNodeWidth(op->body_); width > 1) {
        cur_.emplace_back(makeIntConst(-1));
        pushFor(op);
        (*this)(op->body_);
        popFor(op);
        cur_.pop_back();
    } else {
        pushFor(op);
        (*this)(op->body_);
        popFor(op);
    }
    cur_.pop_back();
    conds_.resize(oldCondsSize);
}

void FindAccessPoint::visit(const If &op) {
    (*this)(op->cond_);

    if (!op->elseCase_.isValid()) {
        conds_.emplace_back(op->cond_);
        (*this)(op->thenCase_);
        conds_.pop_back();
    } else {
        conds_.emplace_back(op->cond_);
        (*this)(op->thenCase_);
        conds_.pop_back();
        conds_.emplace_back(makeLNot(op->cond_));
        (*this)(op->elseCase_);
        conds_.pop_back();
    }
}

void FindAccessPoint::visit(const Load &op) {
    if (!cur_.empty() &&
        cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
        // top is band node
        if (!lastIsLoad_) {
            cur_.back().iter_ =
                makeIntConst(cur_.back().iter_.as<IntConstNode>()->val_ + 1);
        }
    }
    lastIsLoad_ = true;

    BaseClass::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op,
           curStmt(),
           def(op->var_),
           buffer(op->var_),
           defAxis_.at(op->var_),
           cur_,
           std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
           conds_,
           symbolTableSnapshot()};
    reads_[def(op->var_)->id()].emplace_back(ap);
}

std::string AnalyzeDeps::makeIterList(const std::vector<IterAxis> &list,
                                      int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret += mangle(list[i].iter_.as<VarNode>()->name_);
            } else if (list[i].iter_->nodeType() == ASTNodeType::IntConst) {
                ret += std::to_string(list[i].iter_.as<IntConstNode>()->val_);
            } else {
                ASSERT(false);
            }
        } else {
            ret += "0";
        }
        if (i < n - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

Ref<std::string> AnalyzeDeps::makeAccList(GenPBExpr &genPBExpr,
                                          const std::vector<Expr> &list,
                                          RelaxMode relax,
                                          GenPBExpr::VarMap &externals) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genPBExpr.gen(list[i]); linstr.isValid()) {
            ret += *linstr;
            for (auto &&[expr, str] : genPBExpr.vars(list[i])) {
                if (expr->nodeType() == ASTNodeType::Load) {
                    externals[expr] = str;
                }
            }
        } else if (relax == RelaxMode::Possible) {
            ret += mangle("free" + std::to_string(i));
        } else {
            return nullptr;
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return Ref<std::string>::make("[" + ret + "]");
}

Ref<std::string> AnalyzeDeps::makeCond(GenPBExpr &genPBExpr,
                                       const std::vector<Expr> &conds,
                                       RelaxMode relax,
                                       GenPBExpr::VarMap &externals) {
    std::string ret;
    for (auto &&cond : conds) {
        if (auto str = genPBExpr.gen(cond); str.isValid()) {
            for (auto &&[expr, str] : genPBExpr.vars(cond)) {
                if (expr->nodeType() == ASTNodeType::Load) {
                    externals[expr] = str;
                }
            }
            if (!ret.empty()) {
                ret += " and ";
            }
            ret += *str;
        } else {
            if (relax == RelaxMode::Necessary) {
                return nullptr;
            } else {
                // Create a dummy integer variable because ISL does not bool
                // variables
                if (cond->nodeType() == ASTNodeType::LNot) {
                    auto predicate =
                        "__pred_" +
                        std::to_string(cond.as<LNotNode>()->expr_->hash()) +
                        genPBExpr.varSuffix();
                    externals[cond.as<LNotNode>()->expr_] = predicate;
                    if (!ret.empty()) {
                        ret += " and ";
                    }
                    ret += predicate + " <= 0";
                } else {
                    auto predicate = "__pred_" + std::to_string(cond->hash()) +
                                     genPBExpr.varSuffix();
                    externals[cond] = predicate;
                    if (!ret.empty()) {
                        ret += " and ";
                    }
                    ret += predicate + " > 0";
                }
            }
        }
    }
    return Ref<std::string>::make(ret);
}

PBMap AnalyzeDeps::makeAccMap(PBCtx &presburger, const AccessPoint &p,
                              int iterDim, int accDim, RelaxMode relax,
                              const std::string &extSuffix,
                              GenPBExpr::VarMap &externals) {
    GenPBExpr genPBExpr(p.symbolTable_, extSuffix);
    auto ret = makeIterList(p.iter_, iterDim) + " -> ";
    if (auto str = makeAccList(genPBExpr, p.access_, relax, externals);
        str.isValid()) {
        ret += *str;
    } else {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, accDim));
    }
    std::string cond;
    if (auto str = makeCond(genPBExpr, p.conds_, relax, externals);
        str.isValid()) {
        cond += (cond.empty() || str->empty() ? "" : " and ") + *str;
    } else {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, accDim));
    }
    if (!cond.empty()) {
        ret += ": " + cond;
    }
    std::string ext;
    if (!externals.empty()) {
        bool first = true;
        for (auto &&[expr, str] : externals) {
            ext += (first ? "" : ", ") + str;
            first = false;
        }
        ext = "[" + ext + "] -> ";
    }
    ret = ext + "{" + ret + "}";
    return PBMap(presburger, ret);
}

std::string AnalyzeDeps::makeNdList(const std::string &name, int n) const {
    std::string ret;
    for (int i = 0; i < n; i++) {
        ret += name + std::to_string(i);
        if (i < n - 1) {
            ret += ",";
        }
    }
    return "[" + ret + "]";
}

PBMap AnalyzeDeps::makeEqForBothOps(
    PBCtx &presburger, const std::vector<std::pair<int, int>> &coord,
    int iterDim) const {
    std::ostringstream os;
    os << "{" << makeNdList("d", iterDim) << " -> " << makeNdList("d_", iterDim)
       << ": ";
    for (auto &&[i, crd] : iter::enumerate(coord)) {
        if (i > 0) {
            os << " and ";
        }
        os << "d" << crd.first << " = " << crd.second << " and "
           << "d_" << crd.first << " = " << crd.second;
    }
    os << "}";
    return PBMap(presburger, os.str());
}

PBMap AnalyzeDeps::makeIneqBetweenOps(PBCtx &presburger, DepDirection mode,
                                      int iterId, int iterDim) const {
    auto idStr = std::to_string(iterId);
    std::string ineq;
    switch (mode) {
    case DepDirection::Inv:
        ineq = ">";
        break;
    case DepDirection::Normal:
        ineq = "<";
        break;
    case DepDirection::Same:
        ineq = "=";
        break;
    case DepDirection::Different:
        ineq = "!=";
        break;
    default:
        ASSERT(false);
    }
    return PBMap(presburger, "{" + makeNdList("d", iterDim) + " -> " +
                                 makeNdList("d_", iterDim) + ": d_" + idStr +
                                 " " + ineq + " d" + idStr + "}");
}

PBMap AnalyzeDeps::makeConstraintOfSingleLoop(PBCtx &presburger, const ID &loop,
                                              DepDirection mode, int iterDim) {
    auto &&coord = scope2coord_.at(loop);
    int iterId = coord.size() - 1;
    if (iterId >= iterDim) {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }

    auto ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));

    // Position in the outer StmtSeq nodes
    std::vector<std::pair<int, int>> pos;
    for (int i = 0; i < iterId; i++) {
        if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
            pos.emplace_back(i, coord[i].iter_.as<IntConstNode>()->val_);
        }
    }
    if (!pos.empty()) {
        ret = intersect(std::move(ret),
                        makeEqForBothOps(presburger, pos, iterDim));
    }

    return intersect(std::move(ret),
                     makeIneqBetweenOps(presburger, mode, iterId, iterDim));
}

PBMap AnalyzeDeps::makeConstraintOfParallelScope(PBCtx &presburger,
                                                 const ParallelScope &parallel,
                                                 DepDirection mode, int iterDim,
                                                 const AccessPoint &later,
                                                 const AccessPoint &earlier) {
    int laterDim = -1, earlierDim = -1;
    for (int i = (int)later.iter_.size() - 1; i >= 0; i--) {
        if (later.iter_[i].parallel_ == parallel) {
            laterDim = i;
            break;
        }
    }
    for (int i = (int)earlier.iter_.size() - 1; i >= 0; i--) {
        if (earlier.iter_[i].parallel_ == parallel) {
            earlierDim = i;
            break;
        }
    }
    if (earlierDim == -1 && laterDim == -1) {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }
    if (earlierDim == -1 || laterDim == -1) {
        return universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }

    std::string ineq;
    switch (mode) {
    case DepDirection::Inv:
        ineq = ">";
        break;
    case DepDirection::Normal:
        ineq = "<";
        break;
    case DepDirection::Same:
        ineq = "=";
        break;
    case DepDirection::Different:
        ineq = "!=";
        break;
    default:
        ASSERT(false);
    }
    // FIXME: parallel loop of the same parallel scope of later and earlier may
    // have different `begin`, we must substract `begin` before compareing
    return PBMap(presburger, "{" + makeNdList("d", iterDim) + " -> " +
                                 makeNdList("d_", iterDim) + ": d_" +
                                 std::to_string(earlierDim) + " " + ineq +
                                 " d" + std::to_string(laterDim) + "}");
}

PBMap AnalyzeDeps::makeExternalEq(PBCtx &presburger, int iterDim,
                                  const std::string &ext1,
                                  const std::string &ext2) {
    std::string mapping =
        makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim);
    return PBMap(presburger, "[" + ext1 + ", " + ext2 + "] -> {" + mapping +
                                 ": " + ext1 + " = " + ext2 + "}");
}

const std::string &AnalyzeDeps::getVar(const AST &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Load:
        return op.as<LoadNode>()->var_;
    case ASTNodeType::Store:
        return op.as<StoreNode>()->var_;
    case ASTNodeType::ReduceTo:
        return op.as<ReduceToNode>()->var_;
    default:
        ASSERT(false);
    }
}

PBMap AnalyzeDeps::makeSerialToAll(PBCtx &presburger, int iterDim,
                                   const std::vector<IterAxis> &point) const {
    std::string to = makeNdList("d", iterDim), from;
    for (int i = 0; i < iterDim; i++) {
        if (i < (int)point.size() && point[i].parallel_ != serialScope) {
            from += std::string(i > 0 ? ", " : "") + "0";
        } else {
            from += std::string(i > 0 ? ", " : "") + "d" + std::to_string(i);
        }
    }
    from = "[" + from + "]";
    return PBMap(presburger, "{" + from + " -> " + to + "}");
}

PBMap AnalyzeDeps::makeEraseVarDefConstraint(PBCtx &presburger,
                                             const Ref<AccessPoint> &point,
                                             int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    if (eraseOutsideVarDef_) {
        for (int i = 0; i < point->defAxis_; i++) {
            ret = intersect(
                std::move(ret),
                makeIneqBetweenOps(presburger, DepDirection::Same, i, iterDim));
        }
    }
    return ret;
}

PBMap AnalyzeDeps::makeNoDepsConstraint(PBCtx &presburger,
                                        const std::string &var, int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    if (noDepsLists_.count(var)) {
        for (auto &&noDepsLoop : noDepsLists_.at(var)) {
            auto noDep = makeConstraintOfSingleLoop(
                presburger, noDepsLoop, DepDirection::Different, iterDim);
            ret = subtract(std::move(ret), std::move(noDep));
        }
    }
    return ret;
}

PBMap AnalyzeDeps::makeExternalVarConstraint(
    PBCtx &presburger, const Ref<AccessPoint> &later,
    const Ref<AccessPoint> &earlier, const GenPBExpr::VarMap &laterExternals,
    const GenPBExpr::VarMap &earlierExternals, int iterDim) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    // We only have to add constraint for common loops of both accesses
    auto common = lcaStmt(later->stmt_, earlier->stmt_);

    for (auto &&[expr, strs] : intersect(laterExternals, earlierExternals)) {
        auto &&[pStr, oStr] = strs;
        // If all of the loops are variant, we don't have to make the constraint
        // at all. This will save time for Presburger solver
        for (auto c = common; c.isValid(); c = c->parentStmt()) {
            if (c->nodeType() == ASTNodeType::For) {
                if (isVariant(variantExpr_, expr, c->id())) {
                    goto found;
                }
                goto do_compute_constraint;
            found:;
            }
        }
        continue;

        // Compute the constraint
    do_compute_constraint:
        auto require = makeExternalEq(presburger, iterDim, pStr, oStr);
        for (auto c = common; c.isValid(); c = c->parentStmt()) {
            if (c->nodeType() == ASTNodeType::For) {
                if (isVariant(variantExpr_, expr, c->id())) {
                    // Since idx[i] must be inside loop i, we only have
                    // to call makeIneqBetweenOps, but no need to call
                    // makeConstraintOfSingleLoop
                    auto diffIter = makeIneqBetweenOps(
                        presburger, DepDirection::Different,
                        scope2coord_.at(c->id()).size() - 1, iterDim);
                    require = uni(std::move(diffIter), std::move(require));
                }
            }
        }
        ret = intersect(std::move(ret), std::move(require));
    }
    return ret;
}

PBMap AnalyzeDeps::projectOutPrivateAxis(PBCtx &presburger, int iterDim,
                                         int since) {
    std::string from = makeNdList("d", iterDim);
    std::string to;
    for (int i = 0; i < iterDim; i++) {
        to += (i > 0 ? ", " : "") + (i < since ? "d" + std::to_string(i) : "0");
    }
    to = "[" + to + "]";
    return PBMap(presburger, "{" + from + " -> " + to + "}");
}

void AnalyzeDeps::projectOutPrivateAxis(
    PBCtx &presburger, const Ref<AccessPoint> &point,
    const std::vector<Ref<AccessPoint>> &otherList,
    std::vector<PBMap> &otherMapList, int iterDim) {
    if (!noProjectOutProvateAxis_) {
        std::vector<int> oCommonDims(otherList.size(), 0);
        for (size_t i = 0, n = otherList.size(); i < n; i++) {
            auto &&other = otherList[i];
            int cpo = numCommonDims(point, other);
            oCommonDims[i] = std::max(oCommonDims[i], cpo);
            if (i + 1 < n) {
                int co1o2 = numCommonDims(other, otherList[i + 1]);
                oCommonDims[i] = std::max(oCommonDims[i], co1o2);
                oCommonDims[i + 1] = std::max(oCommonDims[i + 1], co1o2);
            }
        }

        for (auto &&[common, other, otherMap] :
             iter::zip(oCommonDims, otherList, otherMapList)) {
            if (common + 1 < (int)other->iter_.size()) {
                otherMap = applyDomain(
                    std::move(otherMap),
                    projectOutPrivateAxis(presburger, iterDim, common + 1));
                otherMap = coalesce(std::move(otherMap));
            }
        }
    }
}

int AnalyzeDeps::numCommonDims(const Ref<AccessPoint> &p1,
                               const Ref<AccessPoint> &p2) {
    int n = std::min(p1->iter_.size(), p2->iter_.size());
    for (int i = 0; i < n; i++) {
        auto &&iter1 = p1->iter_[i].iter_;
        auto &&iter2 = p2->iter_[i].iter_;
        if (iter1->nodeType() == ASTNodeType::IntConst &&
            iter2->nodeType() == ASTNodeType::IntConst &&
            iter1.as<IntConstNode>()->val_ != iter2.as<IntConstNode>()->val_) {
            for (int j = n - 1; j >= i; j--) {
                if ((j < (int)p1->iter_.size() &&
                     p1->iter_[j].parallel_ != serialScope) ||
                    (j < (int)p2->iter_.size() &&
                     p2->iter_[j].parallel_ != serialScope)) {
                    return j + 1;
                }
            }
            return i;
        }
    }
    return n;
}

void AnalyzeDeps::checkAgainstCond(PBCtx &presburger,
                                   const Ref<AccessPoint> &later,
                                   const Ref<AccessPoint> &earlier,
                                   const PBMap &depAll, const PBMap &nearest,
                                   const PBMap &laterMap,
                                   const PBMap &earlierMap, int iterDim) {
    if (nearest.empty()) {
        return;
    }
    // FIXME: Should these killing tests be after the conditional checks, for
    // correctness?
    if ((mode_ == FindDepsMode::KillEarlier ||
         mode_ == FindDepsMode::KillBoth) &&
        domain(earlierMap) != range(nearest)) {
        return;
    }
    if ((mode_ == FindDepsMode::KillLater || mode_ == FindDepsMode::KillBoth) &&
        domain(laterMap) != domain(nearest)) {
        return;
    }

    for (auto &&item : cond_) {
        std::vector<PBMap> _requires;
        for (auto &&[nodeOrParallel, dir] : item) {
            if (nodeOrParallel.isNode_) {
                _requires.emplace_back(makeConstraintOfSingleLoop(
                    presburger, nodeOrParallel.id_, dir, iterDim));
            } else {
                _requires.emplace_back(makeConstraintOfParallelScope(
                    presburger, nodeOrParallel.parallel_, dir, iterDim, *later,
                    *earlier));
            }
        }

        // Early exit: if there is no intersection on `depAll`, there must
        // be no intersection on `nearest`. Computing on `nearest` is much
        // heavier because it contains more basic maps
        PBMap res = nearest, possible = depAll;
        for (auto &&require : _requires) {
            possible = intersect(std::move(possible), require);
            if (possible.empty()) {
                goto fail;
            }
        }

        for (auto &&require : _requires) {
            res = intersect(std::move(res), std::move(require));
            if (res.empty()) {
                goto fail;
            }
        }
        {
            std::lock_guard<std::mutex> guard(lock_);
            if (noProjectOutProvateAxis_) {
                found_(Dependency{item, getVar(later->op_), *later, *earlier,
                                  iterDim, res, laterMap, earlierMap,
                                  presburger, *this});
            } else {
                // It will be misleading if we pass Presburger maps to users in
                // this case
                found_(Dependency{item, getVar(later->op_), *later, *earlier,
                                  iterDim, PBMap(), PBMap(), PBMap(),
                                  presburger, *this});
            }
        }
    fail:;
    }
}

void AnalyzeDeps::checkDepLatestEarlier(
    const Ref<AccessPoint> &later,
    const std::vector<Ref<AccessPoint>> &_earlierList) {
    std::vector<Ref<AccessPoint>> earlierList;
    for (auto &&earlier : _earlierList) {
        if (ignoreReductionWAW_ &&
            later->op_->nodeType() == ASTNodeType::ReduceTo &&
            earlier->op_->nodeType() == ASTNodeType::ReduceTo) {
            continue;
        }
        if (filter_ == nullptr || filter_(*later, *earlier)) {
            earlierList.emplace_back(earlier);
        }
    }
    if (earlierList.empty()) {
        return;
    }
    tasks_.emplace_back([later, earlierList = std::move(earlierList), this]() {
        PBCtx presburger;
        checkDepLatestEarlierImpl(presburger, later, earlierList);
    });
}

void AnalyzeDeps::checkDepEarliestLater(
    const std::vector<Ref<AccessPoint>> &_laterList,
    const Ref<AccessPoint> &earlier) {
    std::vector<Ref<AccessPoint>> laterList;
    for (auto &&later : _laterList) {
        if (ignoreReductionWAW_ &&
            later->op_->nodeType() == ASTNodeType::ReduceTo &&
            earlier->op_->nodeType() == ASTNodeType::ReduceTo) {
            continue;
        }
        if (filter_ == nullptr || filter_(*later, *earlier)) {
            laterList.emplace_back(later);
        }
    }
    if (laterList.empty()) {
        return;
    }
    tasks_.emplace_back([laterList = std::move(laterList), earlier, this]() {
        PBCtx presburger;
        checkDepEarliestLaterImpl(presburger, laterList, earlier);
    });
}

void AnalyzeDeps::checkDepLatestEarlierImpl(
    PBCtx &presburger, const Ref<AccessPoint> &later,
    const std::vector<Ref<AccessPoint>> &earlierList) {
    int accDim = later->access_.size();
    int iterDim = later->iter_.size();
    for (auto &&earlier : earlierList) {
        iterDim = std::max<int>(iterDim, earlier->iter_.size());
        ASSERT((int)earlier->access_.size() == accDim);
    }

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, later, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, later->def_->name_, iterDim);

    GenPBExpr::VarMap laterExternals;
    PBMap laterMap = makeAccMap(presburger, *later, iterDim, accDim,
                                laterRelax_, "__ext_later", laterExternals);
    if (laterMap.empty()) {
        return;
    }
    PBMap ls2a = makeSerialToAll(presburger, iterDim, later->iter_);
    PBMap la2s = reverse(ls2a);
    std::vector<PBMap> earlierMapList(earlierList.size());
    std::vector<GenPBExpr::VarMap> earlierExternalsList(earlierList.size());
    std::vector<PBMap> es2aList(earlierList.size()),
        depAllList(earlierList.size());
    PBMap psDepAllUnion;
    for (auto &&[i, earlier, earlierMap, earlierExternals] :
         iter::zip(iter::count(), earlierList, earlierMapList,
                   earlierExternalsList)) {
        earlierMap =
            makeAccMap(presburger, *earlier, iterDim, accDim, earlierRelax_,
                       "__ext_earlier" + std::to_string(i), earlierExternals);
    }
    projectOutPrivateAxis(presburger, later, earlierList, earlierMapList,
                          iterDim);
    for (auto &&[i, earlier, earlierMap, earlierExternals, es2a, depAll] :
         iter::zip(iter::count(), earlierList, earlierMapList,
                   earlierExternalsList, es2aList, depAllList)) {
        if (earlierMap.empty()) {
            continue;
        }
        es2a = makeSerialToAll(presburger, iterDim, earlier->iter_);
        PBMap ea2s = reverse(es2a);

        depAll = subtract(applyRange(laterMap, reverse(earlierMap)), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll = intersect(std::move(depAll),
                           makeExternalVarConstraint(
                               presburger, later, earlier, laterExternals,
                               earlierExternals, iterDim));
        depAll = coalesce(std::move(depAll));

        PBMap psDepAll = applyRange(depAll, std::move(ea2s));
        psDepAllUnion = psDepAllUnion.isValid()
                            ? uni(std::move(psDepAllUnion), std::move(psDepAll))
                            : std::move(psDepAll);
    }
    if (!psDepAllUnion.isValid()) {
        return;
    }

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(std::move(ls2a), psDepAllUnion);
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap psDep = intersect(applyRange(la2s, std::move(ssDep)), psDepAllUnion);
    PBMap psSelf = intersect(applyRange(std::move(la2s), std::move(ssSelf)),
                             std::move(psDepAllUnion));
    PBMap psNearest = uni(lexmax(std::move(psDep)), std::move(psSelf));
    psNearest = coalesce(std::move(psNearest));

    for (auto &&[earlier, es2a, earlierMap, depAll] :
         iter::zip(earlierList, es2aList, earlierMapList, depAllList)) {
        if (depAll.isValid()) {
            checkAgainstCond(
                presburger, later, earlier, depAll,
                intersect(applyRange(psNearest, std::move(es2a)), depAll),
                laterMap, earlierMap, iterDim);
        }
    }
}

void AnalyzeDeps::checkDepEarliestLaterImpl(
    PBCtx &presburger, const std::vector<Ref<AccessPoint>> &laterList,
    const Ref<AccessPoint> &earlier) {
    int accDim = earlier->access_.size();
    int iterDim = earlier->iter_.size();
    for (auto &&later : laterList) {
        iterDim = std::max<int>(iterDim, later->iter_.size());
        ASSERT((int)later->access_.size() == accDim);
    }

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, earlier, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, earlier->def_->name_, iterDim);

    GenPBExpr::VarMap earlierExternals;
    PBMap earlierMap =
        makeAccMap(presburger, *earlier, iterDim, accDim, earlierRelax_,
                   "__ext_earlier", earlierExternals);
    if (earlierMap.empty()) {
        return;
    }
    PBMap es2a = makeSerialToAll(presburger, iterDim, earlier->iter_);
    PBMap ea2s = reverse(es2a);
    std::vector<PBMap> laterMapList(laterList.size());
    std::vector<GenPBExpr::VarMap> laterExternalsList(laterList.size());
    std::vector<PBMap> ls2aList(laterList.size()), depAllList(laterList.size());
    PBMap spDepAllUnion;
    for (auto &&[i, later, laterMap, laterExternals] : iter::zip(
             iter::count(), laterList, laterMapList, laterExternalsList)) {
        laterMap =
            makeAccMap(presburger, *later, iterDim, accDim, laterRelax_,
                       "__ext_later" + std::to_string(i), laterExternals);
    }
    projectOutPrivateAxis(presburger, earlier, laterList, laterMapList,
                          iterDim);
    for (auto &&[i, later, laterMap, laterExternals, ls2a, depAll] :
         iter::zip(iter::count(), laterList, laterMapList, laterExternalsList,
                   ls2aList, depAllList)) {
        if (laterMap.empty()) {
            continue;
        }
        ls2a = makeSerialToAll(presburger, iterDim, later->iter_);
        PBMap la2s = reverse(ls2a);

        depAll = subtract(applyRange(laterMap, reverse(earlierMap)), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll = intersect(std::move(depAll),
                           makeExternalVarConstraint(
                               presburger, later, earlier, laterExternals,
                               earlierExternals, iterDim));
        depAll = coalesce(std::move(depAll));

        PBMap spDepAll = applyDomain(depAll, std::move(la2s));
        spDepAllUnion = spDepAllUnion.isValid()
                            ? uni(std::move(spDepAllUnion), std::move(spDepAll))
                            : std::move(spDepAll);
    }
    if (!spDepAllUnion.isValid()) {
        return;
    }

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(spDepAllUnion, std::move(ea2s));
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap spDep = intersect(applyRange(std::move(ssDep), es2a), spDepAllUnion);
    PBMap spSelf = intersect(applyRange(std::move(ssSelf), std::move(es2a)),
                             std::move(spDepAllUnion));
    PBMap spNearest =
        uni(reverse(lexmin(reverse(std::move(spDep)))), std::move(spSelf));
    spNearest = coalesce(std::move(spNearest));

    for (auto &&[later, ls2a, laterMap, depAll] :
         iter::zip(laterList, ls2aList, laterMapList, depAllList)) {
        if (depAll.isValid()) {
            checkAgainstCond(
                presburger, later, earlier, depAll,
                intersect(applyDomain(spNearest, std::move(ls2a)), depAll),
                laterMap, earlierMap, iterDim);
        }
    }
}

void AnalyzeDeps::genTasks() {
    for (auto &&def : allDefs_) {
        if (writes_.count(def->id())) {
            auto &&allWrites = writes_.at(def->id());
            if (reads_.count(def->id())) {
                for (auto &&read : reads_.at(def->id())) {
                    if (depType_ & DEP_RAW) {
                        checkDepLatestEarlier(read, allWrites);
                    }
                    if (depType_ & DEP_WAR) {
                        checkDepEarliestLater(allWrites, read);
                    }
                }
            }

            for (auto &&write : allWrites) {
                // Store    -> Store    : WAW
                // ReduceTo -> Store    : WAW, WAR
                // Store    -> ReduceTo : WAW, RAW
                // ReduceTo -> ReduceTo : WAW, RAW, WAR
                if (depType_ & DEP_WAW) {
                    // Every Store checks its immediate predecessor, so we do
                    // not have to check its follower
                    checkDepLatestEarlier(write, allWrites);
                } else if (write->op_->nodeType() == ASTNodeType::ReduceTo) {
                    if (depType_ & DEP_RAW) {
                        checkDepLatestEarlier(write, allWrites);
                    }
                    if (depType_ & DEP_WAR) {
                        checkDepEarliestLater(allWrites, write);
                    }
                }
            }
        }
    }
}

PBMap Dependency::extraCheck(PBMap dep,
                             const NodeIDOrParallelScope &nodeOrParallel,
                             const DepDirection &dir) const {
    PBMap require;
    if (nodeOrParallel.isNode_) {
        require = self_.makeConstraintOfSingleLoop(
            presburger_, nodeOrParallel.id_, dir, iterDim_);
    } else {
        require = self_.makeConstraintOfParallelScope(
            presburger_, nodeOrParallel.parallel_, dir, iterDim_, later_,
            earlier_);
    }
    dep = intersect(std::move(dep), std::move(require));
    return dep;
}

void findDeps(const Stmt &op, const std::vector<FindDepsCond> &cond,
              const FindDepsCallback &found, FindDepsMode mode, DepType depType,
              const FindDepsFilter &filter, bool ignoreReductionWAW,
              bool eraseOutsideVarDef, bool noProjectOutProvateAxis) {
    if (cond.empty()) {
        return;
    }

    if (mode != FindDepsMode::Dep) {
        noProjectOutProvateAxis = true;
    }

    FindAccessPoint accFinder(op);
    accFinder(op);
    FindAllNoDeps noDepsFinder;
    noDepsFinder(op);
    auto variantExpr = findLoopVariance(op).first;
    AnalyzeDeps analyzer(accFinder.reads(), accFinder.writes(),
                         accFinder.allDefs(), accFinder.scope2coord(),
                         noDepsFinder.results(), variantExpr, cond, found, mode,
                         depType, filter, ignoreReductionWAW,
                         eraseOutsideVarDef, noProjectOutProvateAxis);
    analyzer.genTasks();
    size_t n = analyzer.tasks().size();
    std::vector<std::exception_ptr> exceptions(n, nullptr);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++) {
        try {
            analyzer.tasks()[i]();
        } catch (...) {
            exceptions[i] = std::current_exception();
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (exceptions[i]) {
            std::rethrow_exception(exceptions[i]);
        }
    }
}

std::string toString(const Dependency &dep) {
    std::ostringstream os;
    os << "Dependency ";
    os << (dep.later()->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << dep.later();
    if (dep.later()->isExpr()) {
        os << " in " << dep.later_.stmt_;
    }
    os << " after ";
    os << (dep.earlier()->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << dep.earlier();
    if (dep.earlier()->isExpr()) {
        os << " in " << dep.earlier_.stmt_;
    }
    bool first = true;
    for (auto &&[scope, dir] : dep.cond_) {
        os << (first ? " along " : " and ");
        first = false;
        if (scope.isNode_) {
            os << toString(scope.id_);
        } else {
            os << toString(scope.parallel_);
        }
    }
    return std::regex_replace(os.str(), std::regex("\n"), "");
}

} // namespace freetensor
