#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
#include <except.h>
#include <mutator.h>
#include <pass/simplify.h>

namespace ir {

template <class T, class V>
static void unionTo(std::unordered_map<T, V> &target,
                    const std::unordered_map<T, V> &other) {
    target.insert(other.begin(), other.end());
}

template <class T, class V>
static std::unordered_map<T, V> intersect(const std::unordered_map<T, V> &lhs,
                                          const std::unordered_map<T, V> &rhs) {
    std::unordered_map<T, V> ret;
    for (auto &&[key, value] : lhs) {
        if (rhs.count(key)) {
            ret.emplace(key, value);
        }
    }
    return ret;
}

static std::string replaceAll(const std::string &str,
                              const std::string &toSearch,
                              const std::string &replaceStr) {
    auto data = str;
    size_t pos = data.find(toSearch);
    while (pos != std::string::npos) {
        data.replace(pos, toSearch.size(), replaceStr);
        pos = data.find(toSearch, pos + replaceStr.size());
    }
    return data;
}

void FindAllNoDeps::visit(const For &op) {
    Visitor::visit(op);
    for (auto &&var : op->property_.noDeps_) {
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
    ASSERT(!defs_.count(op->name_));
    allDefs_.emplace_back(op);
    defAxis_[op->name_] =
        !cur_.empty() && cur_.back().iter_->nodeType() == ASTNodeType::IntConst
            ? cur_.size() - 1
            : cur_.size();
    defs_[op->name_] = op;
    Visitor::visit(op);
    defAxis_.erase(op->name_);
    defs_.erase(op->name_);
}

void FindAccessPoint::visit(const StmtSeq &op) {
    if (!cur_.empty() &&
        cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
        scope2coord_[op->id()] = cur_;
    }
    Visitor::visit(op);
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
    auto rbegin = makeAdd(
        op->begin_, makeMul(makeSub(op->len_, makeIntConst(1)), op->step_));
    conds_.emplace_back(makeGE(iter, makeMin(op->begin_, rbegin)));
    conds_.emplace_back(makeLE(iter, makeMax(rbegin, op->begin_)));
    cur_.emplace_back(iter, op->property_.parallel_);
    scope2coord_[op->id()] = cur_;
    if (int width = countBandNodeWidth(op->body_); width > 1) {
        cur_.emplace_back(makeIntConst(-1));
        (*this)(op->body_);
        cur_.pop_back();
    } else {
        (*this)(op->body_);
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

    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op,
           cursor(),
           defs_.at(op->var_),
           defs_.at(op->var_)->buffer_,
           defAxis_.at(op->var_),
           cur_,
           std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
           conds_};
    reads_[defs_.at(op->var_)->id()].emplace_back(ap);
}

void GenPBExprDeps::visitExpr(const Expr &op) {
    auto oldParent = parent_;
    parent_ = op;
    GenPBExpr::visitExpr(op);
    parent_ = oldParent;
    if (parent_.isValid()) {
        unionTo(externals_[parent_], externals_[op]);
    }
}

void GenPBExprDeps::visit(const Load &op) {
    getHash_(op);
    auto h = getHash_.hash().at(op);
    auto str = normalizeId("ext" + std::to_string(h)) + "!!placeholder!!";
    externals_[op][h] = std::make_pair(op, str);
    results_[op] = str;
}

std::string AnalyzeDeps::makeIterList(GenPBExprDeps &genPBExpr,
                                      const std::vector<IterAxis> &list,
                                      int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret +=
                    genPBExpr.normalizeId(list[i].iter_.as<VarNode>()->name_);
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

Ref<std::string> AnalyzeDeps::makeAccList(GenPBExprDeps &genPBExpr,
                                          const std::vector<Expr> &list,
                                          RelaxMode relax,
                                          ExternalMap &externals) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genPBExpr.gen(list[i]); linstr.isValid()) {
            ret += *linstr;
            unionTo(externals, genPBExpr.externals(list[i]));
        } else if (relax == RelaxMode::Possible) {
            ret += genPBExpr.normalizeId("free" + std::to_string(i));
        } else {
            return nullptr;
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return Ref<std::string>::make("[" + ret + "]");
}

Ref<std::string> AnalyzeDeps::makeCond(GenPBExprDeps &genPBExpr,
                                       const std::vector<Expr> &conds,
                                       RelaxMode relax,
                                       ExternalMap &externals) {
    std::string ret;
    for (auto &&cond : conds) {
        if (auto str = genPBExpr.gen(cond); str.isValid()) {
            unionTo(externals, genPBExpr.externals(cond));
            if (!ret.empty()) {
                ret += " and ";
            }
            ret += *str;
        } else if (relax == RelaxMode::Necessary) {
            return nullptr;
        }
    }
    return Ref<std::string>::make(ret);
}

PBMap AnalyzeDeps::makeAccMap(PBCtx &presburger, GenPBExprDeps &genPBExpr,
                              const AccessPoint &p, int iterDim, int accDim,
                              RelaxMode relax, const std::string &extSuffix,
                              ExternalMap &externals) {
    auto ret = makeIterList(genPBExpr, p.iter_, iterDim) + " -> ";
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
        for (auto &&[hash, item] : externals) {
            ext += (first ? "" : ", ") + item.second;
            first = false;
        }
        ext = "[" + ext + "] -> ";
    }
    ret = ext + "{" + ret + "}";
    ret = replaceAll(ret, "!!placeholder!!", extSuffix);
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
    for (size_t i = 0, iEnd = coord.size(); i < iEnd; i++) {
        if (i > 0) {
            os << " and ";
        }
        os << "d" << coord[i].first << " = " << coord[i].second << " and "
           << "d_" << coord[i].first << " = " << coord[i].second;
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

PBMap AnalyzeDeps::makeConstraintOfSingleLoop(PBCtx &presburger,
                                              const std::string &loop,
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

PBMap AnalyzeDeps::makeConstraintOfParallelScope(
    PBCtx &presburger, const std::string &parallel, DepDirection mode,
    int iterDim, const Ref<AccessPoint> &point, const Ref<AccessPoint> &other) {
    int pointDim = -1, otherDim = -1;
    for (int i = (int)point->iter_.size() - 1; i >= 0; i--) {
        if (point->iter_[i].parallel_ == parallel) {
            pointDim = i;
            break;
        }
    }
    for (int i = (int)other->iter_.size() - 1; i >= 0; i--) {
        if (other->iter_[i].parallel_ == parallel) {
            otherDim = i;
            break;
        }
    }
    if (otherDim == -1 && pointDim == -1) {
        return emptyMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    }
    if (otherDim == -1 || pointDim == -1) {
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
    // FIXME: parallel loop of the same parallel scope of point and other may
    // have different `begin`, we must substract `begin` before compareing
    return PBMap(presburger, "{" + makeNdList("d", iterDim) + " -> " +
                                 makeNdList("d_", iterDim) + ": d_" +
                                 std::to_string(otherDim) + " " + ineq + " d" +
                                 std::to_string(pointDim) + "}");
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
        if (i < (int)point.size() && !point[i].parallel_.empty()) {
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
    PBCtx &presburger, const Ref<AccessPoint> &point,
    const Ref<AccessPoint> &other, const ExternalMap &pExternals,
    const ExternalMap &oExternals, int iterDim, const std::string &extSuffixP,
    const std::string &extSuffixO) {
    PBMap ret = universeMap(spaceAlloc(presburger, 0, iterDim, iterDim));
    // We only have to add constraint for common loops of both accesses
    auto common = lca(point->cursor_, other->cursor_);

    for (auto &&[hash, item] : intersect(pExternals, oExternals)) {
        // If all of the loops are variant, we don't have to make the constraint
        // at all. This will save time for Presburger solver
        for (auto c = common;; c = c.outer()) {
            if (c.nodeType() == ASTNodeType::For) {
                if (isVariant(variantExpr_, item.first, c.id())) {
                    goto found;
                }
                goto do_compute_constraint;
            found:;
            }
            if (!c.hasOuter()) {
                break;
            }
        }
        continue;

        // Compute the constraint
    do_compute_constraint:
        auto require = makeExternalEq(
            presburger, iterDim,
            replaceAll(item.second, "!!placeholder!!", extSuffixP),
            replaceAll(item.second, "!!placeholder!!", extSuffixO));
        for (auto c = common;; c = c.outer()) {
            if (c.nodeType() == ASTNodeType::For) {
                if (isVariant(variantExpr_, item.first, c.id())) {
                    // Since idx[i] must be inside loop i, we only have
                    // to call makeIneqBetweenOps, but no need to call
                    // makeConstraintOfSingleLoop
                    auto diffIter = makeIneqBetweenOps(
                        presburger, DepDirection::Different,
                        scope2coord_.at(c.id()).size() - 1, iterDim);
                    require = uni(std::move(diffIter), std::move(require));
                }
            }
            if (!c.hasOuter()) {
                break;
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
    const std::vector<Ref<AccessPoint>> &otherList, PBMap &pmap,
    std::vector<PBMap> &omapList, int iterDim) {
    if (mode_ == FindDepsMode::Dep) {
        int pCommonDims = 0;
        std::vector<int> oCommonDims(otherList.size(), 0);
        for (size_t i = 0, n = otherList.size(); i < n; i++) {
            auto &&other = otherList[i];
            int cpo = numCommonDims(point, other);
            pCommonDims = std::max(pCommonDims, cpo);
            oCommonDims[i] = std::max(oCommonDims[i], cpo);
            if (i + 1 < n) {
                int co1o2 = numCommonDims(other, otherList[i + 1]);
                oCommonDims[i] = std::max(oCommonDims[i], co1o2);
                oCommonDims[i + 1] = std::max(oCommonDims[i + 1], co1o2);
            }
        }

        if (pCommonDims + 1 < (int)point->iter_.size()) {
            pmap = applyDomain(
                std::move(pmap),
                projectOutPrivateAxis(presburger, iterDim, pCommonDims + 1));
        }
        for (size_t i = 0, n = otherList.size(); i < n; i++) {
            auto &other = otherList[i];
            auto &omap = omapList[i];
            if (oCommonDims[i] + 1 < (int)other->iter_.size()) {
                omap = applyDomain(std::move(omap),
                                   projectOutPrivateAxis(presburger, iterDim,
                                                         oCommonDims[i] + 1));
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
                     !p1->iter_[j].parallel_.empty()) ||
                    (j < (int)p2->iter_.size() &&
                     !p2->iter_[j].parallel_.empty())) {
                    return j + 1;
                }
            }
            return i;
        }
    }
    return n;
}

void AnalyzeDeps::checkDepLatestEarlier(
    const Ref<AccessPoint> &point,
    const std::vector<Ref<AccessPoint>> &_otherList) {
    std::vector<Ref<AccessPoint>> otherList;
    for (auto &&other : _otherList) {
        if (filter_ == nullptr || filter_(*point, *other)) {
            otherList.emplace_back(other);
        }
    }
    if (otherList.empty()) {
        return;
    }
    tasks_.emplace_back([point, otherList = std::move(otherList), this]() {
        PBCtx presburger;
        GenPBExprDeps genPBExpr;
        checkDepLatestEarlierImpl(presburger, genPBExpr, point, otherList);
    });
}

void AnalyzeDeps::checkDepEarliestLater(
    const std::vector<Ref<AccessPoint>> &_pointList,
    const Ref<AccessPoint> &other) {
    std::vector<Ref<AccessPoint>> pointList;
    for (auto &&point : _pointList) {
        if (filter_ == nullptr || filter_(*point, *other)) {
            pointList.emplace_back(point);
        }
    }
    if (pointList.empty()) {
        return;
    }
    tasks_.emplace_back([pointList = std::move(pointList), other, this]() {
        PBCtx presburger;
        GenPBExprDeps genPBExpr;
        checkDepEarliestLaterImpl(presburger, genPBExpr, pointList, other);
    });
}

void AnalyzeDeps::checkDepLatestEarlierImpl(
    PBCtx &presburger, GenPBExprDeps &genPBExpr, const Ref<AccessPoint> &point,
    const std::vector<Ref<AccessPoint>> &otherList) {
    auto pRelax =
        mode_ == FindDepsMode::KillEarlier || mode_ == FindDepsMode::KillBoth
            ? RelaxMode::Necessary
            : RelaxMode::Possible; // later
    auto oRelax =
        mode_ == FindDepsMode::KillLater || mode_ == FindDepsMode::KillBoth
            ? RelaxMode::Necessary
            : RelaxMode::Possible; // earlier

    int accDim = point->access_.size();
    int iterDim = point->iter_.size();
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        iterDim = std::max<int>(iterDim, other->iter_.size());
        ASSERT((int)other->access_.size() == accDim);
    }

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, point, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, point->def_->name_, iterDim);

    ExternalMap pExternals;
    PBMap pmap = makeAccMap(presburger, genPBExpr, *point, iterDim, accDim,
                            pRelax, "__ext_p", pExternals);
    if (pmap.empty()) {
        return;
    }
    PBMap ps2a = makeSerialToAll(presburger, iterDim, point->iter_);
    PBMap pa2s = reverse(ps2a);
    PBSet pIter = domain(pmap);
    std::vector<PBMap> omapList(otherList.size());
    std::vector<ExternalMap> oExternalsList(otherList.size());
    std::vector<PBMap> os2aList(otherList.size()), depAllList(otherList.size());
    std::vector<PBSet> oIterList(otherList.size());
    PBMap psDepAllUnion;
    std::vector<bool> filteredIn(otherList.size(), true);
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &other = otherList[i];
        auto &omap = omapList[i];
        auto &oExternals = oExternalsList[i];
        omap = makeAccMap(presburger, genPBExpr, *other, iterDim, accDim,
                          oRelax, "__ext_o" + std::to_string(i), oExternals);
        if (omap.empty()) {
            filteredIn[i] = false;
        }
    }
    projectOutPrivateAxis(presburger, point, otherList, pmap, omapList,
                          iterDim);
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &other = otherList[i];
        auto &omap = omapList[i];
        auto &oExternals = oExternalsList[i];
        auto &os2a = os2aList[i];
        auto &oIter = oIterList[i];
        auto &depAll = depAllList[i];
        if (!filteredIn[i]) {
            continue;
        }
        os2a = makeSerialToAll(presburger, iterDim, other->iter_);
        PBMap oa2s = reverse(os2a);
        oIter = domain(omap);

        depAll = subtract(applyRange(pmap, reverse(std::move(omap))), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll =
            intersect(std::move(depAll),
                      makeExternalVarConstraint(
                          presburger, point, other, pExternals, oExternals,
                          iterDim, "__ext_p", "__ext_o" + std::to_string(i)));

        PBMap psDepAll = applyRange(depAll, std::move(oa2s));
        psDepAllUnion = psDepAllUnion.isValid()
                            ? uni(std::move(psDepAllUnion), std::move(psDepAll))
                            : std::move(psDepAll);
    }
    if (!psDepAllUnion.isValid()) {
        return;
    }

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(std::move(ps2a), psDepAllUnion);
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap psDep = intersect(applyRange(pa2s, std::move(ssDep)), psDepAllUnion);
    PBMap psSelf = intersect(applyRange(std::move(pa2s), std::move(ssSelf)),
                             std::move(psDepAllUnion));
    PBMap psNearest = uni(lexmax(std::move(psDep)), std::move(psSelf));

    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (!filteredIn[i]) {
            continue;
        }

        auto &&os2a = os2aList[i];
        auto &&oIter = oIterList[i];
        auto &&depAll = depAllList[i];
        PBMap nearest = intersect(applyRange(psNearest, std::move(os2a)),
                                  std::move(depAll));

        if (nearest.empty()) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillEarlier ||
             mode_ == FindDepsMode::KillBoth) &&
            oIter != range(nearest)) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillLater ||
             mode_ == FindDepsMode::KillBoth) &&
            pIter != domain(nearest)) {
            continue;
        }

        for (auto &&item : cond_) {
            PBMap res = nearest;
            bool fail = false;
            for (auto &&[nodeOrParallel, dir] : item) {
                PBMap require;
                if (nodeOrParallel.isNode_) {
                    require = makeConstraintOfSingleLoop(
                        presburger, nodeOrParallel.name_, dir, iterDim);
                } else {
                    require = makeConstraintOfParallelScope(
                        presburger, nodeOrParallel.name_, dir, iterDim, point,
                        other);
                }
                res = intersect(std::move(res), std::move(require));
                if (res.empty()) {
                    fail = true;
                    break;
                }
            }
            if (!fail) {
                std::lock_guard<std::mutex> guard(lock_);
                found_(Dependency{item, getVar(point->op_), *point, *other});
            }
        }
    }
}

void AnalyzeDeps::checkDepEarliestLaterImpl(
    PBCtx &presburger, GenPBExprDeps &genPBExpr,
    const std::vector<Ref<AccessPoint>> &pointList,
    const Ref<AccessPoint> &other) {
    auto pRelax =
        mode_ == FindDepsMode::KillEarlier || mode_ == FindDepsMode::KillBoth
            ? RelaxMode::Necessary
            : RelaxMode::Possible; // later
    auto oRelax =
        mode_ == FindDepsMode::KillLater || mode_ == FindDepsMode::KillBoth
            ? RelaxMode::Necessary
            : RelaxMode::Possible; // earlier

    int accDim = other->access_.size();
    int iterDim = other->iter_.size();
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        iterDim = std::max<int>(iterDim, point->iter_.size());
        ASSERT((int)point->access_.size() == accDim);
    }

    PBMap allEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(presburger, other, iterDim);
    PBMap noDepsConstraint =
        makeNoDepsConstraint(presburger, other->def_->name_, iterDim);

    ExternalMap oExternals;
    PBMap omap = makeAccMap(presburger, genPBExpr, *other, iterDim, accDim,
                            oRelax, "__ext_o", oExternals);
    if (omap.empty()) {
        return;
    }
    PBMap os2a = makeSerialToAll(presburger, iterDim, other->iter_);
    PBMap oa2s = reverse(os2a);
    PBSet oIter = domain(omap);
    std::vector<PBMap> pmapList(pointList.size());
    std::vector<ExternalMap> pExternalsList(pointList.size());
    std::vector<PBMap> ps2aList(pointList.size()), depAllList(pointList.size());
    std::vector<PBSet> pIterList(pointList.size());
    PBMap spDepAllUnion;
    std::vector<bool> filteredIn(pointList.size(), true);
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &point = pointList[i];
        auto &pmap = pmapList[i];
        auto &pExternals = pExternalsList[i];
        pmap = makeAccMap(presburger, genPBExpr, *point, iterDim, accDim,
                          pRelax, "__ext_p" + std::to_string(i), pExternals);
        if (pmap.empty()) {
            filteredIn[i] = false;
        }
    }
    projectOutPrivateAxis(presburger, other, pointList, omap, pmapList,
                          iterDim);
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &point = pointList[i];
        auto &pmap = pmapList[i];
        auto &pExternals = pExternalsList[i];
        auto &ps2a = ps2aList[i];
        auto &pIter = pIterList[i];
        auto &depAll = depAllList[i];
        if (!filteredIn[i]) {
            continue;
        }
        ps2a = makeSerialToAll(presburger, iterDim, point->iter_);
        PBMap pa2s = reverse(ps2a);
        pIter = domain(pmap);

        depAll = subtract(applyRange(std::move(pmap), reverse(omap)), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll =
            intersect(std::move(depAll),
                      makeExternalVarConstraint(
                          presburger, point, other, pExternals, oExternals,
                          iterDim, "__ext_p" + std::to_string(i), "__ext_o"));

        PBMap spDepAll = applyDomain(depAll, std::move(pa2s));
        spDepAllUnion = spDepAllUnion.isValid()
                            ? uni(std::move(spDepAllUnion), std::move(spDepAll))
                            : std::move(spDepAll);
    }
    if (!spDepAllUnion.isValid()) {
        return;
    }

    PBMap serialLexGT = lexGT(spaceSetAlloc(presburger, 0, iterDim));
    PBMap serialEQ = identity(spaceAlloc(presburger, 0, iterDim, iterDim));
    PBMap ssDepAll = applyRange(spDepAllUnion, std::move(oa2s));
    PBMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    PBMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    PBMap spDep = intersect(applyRange(std::move(ssDep), os2a), spDepAllUnion);
    PBMap spSelf = intersect(applyRange(std::move(ssSelf), std::move(os2a)),
                             std::move(spDepAllUnion));
    PBMap spNearest =
        uni(reverse(lexmin(reverse(std::move(spDep)))), std::move(spSelf));

    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        if (!filteredIn[i]) {
            continue;
        }

        auto &&ps2a = ps2aList[i];
        auto &&pIter = pIterList[i];
        auto &&depAll = depAllList[i];
        PBMap nearest = intersect(applyDomain(spNearest, std::move(ps2a)),
                                  std::move(depAll));

        if (nearest.empty()) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillEarlier ||
             mode_ == FindDepsMode::KillBoth) &&
            oIter != range(nearest)) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillLater ||
             mode_ == FindDepsMode::KillBoth) &&
            pIter != domain(nearest)) {
            continue;
        }

        for (auto &&item : cond_) {
            PBMap res = nearest;
            bool fail = false;
            for (auto &&[nodeOrParallel, dir] : item) {
                PBMap require;
                if (nodeOrParallel.isNode_) {
                    require = makeConstraintOfSingleLoop(
                        presburger, nodeOrParallel.name_, dir, iterDim);
                } else {
                    require = makeConstraintOfParallelScope(
                        presburger, nodeOrParallel.name_, dir, iterDim, point,
                        other);
                }
                res = intersect(std::move(res), std::move(require));
                if (res.empty()) {
                    fail = true;
                    break;
                }
            }
            if (!fail) {
                std::lock_guard<std::mutex> guard(lock_);
                found_(Dependency{item, getVar(point->op_), *point, *other});
            }
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
                if (write->op_->nodeType() == ASTNodeType::Store) {
                    if (depType_ & DEP_WAW) {
                        checkDepLatestEarlier(write, allWrites);
                    }

                } else {
                    ASSERT(write->op_->nodeType() == ASTNodeType::ReduceTo);
                    if ((depType_ & DEP_RAW) || (depType_ & DEP_WAW)) {
                        std::vector<Ref<AccessPoint>> others;
                        for (auto &&item : allWrites) {
                            if (ignoreReductionWAW_ &&
                                item->op_->nodeType() ==
                                    ASTNodeType::ReduceTo) {
                                continue;
                            }
                            others.emplace_back(item);
                        }
                        checkDepLatestEarlier(write, others);
                    }

                    if (depType_ & DEP_WAR) {
                        std::vector<Ref<AccessPoint>> points;
                        for (auto &&item : allWrites) {
                            if (ignoreReductionWAW_ &&
                                item->op_->nodeType() ==
                                    ASTNodeType::ReduceTo) {
                                continue;
                            }
                            points.emplace_back(item);
                        }
                        checkDepEarliestLater(points, write);
                    }
                }
            }
        }
    }
}

void findDeps(const Stmt &op, const std::vector<FindDepsCond> &cond,
              const FindDepsCallback &found, FindDepsMode mode, DepType depType,
              const FindDepsFilter &filter, bool ignoreReductionWAW,
              bool eraseOutsideVarDef) {
    if (cond.empty()) {
        return;
    }

    FindAccessPoint accFinder(op);
    accFinder(op);
    FindAllNoDeps noDepsFinder;
    noDepsFinder(op);
    auto variantExpr = findLoopVariance(op).first;
    AnalyzeDeps analyzer(
        accFinder.reads(), accFinder.writes(), accFinder.allDefs(),
        accFinder.scope2coord(), noDepsFinder.results(), variantExpr, cond,
        found, mode, depType, filter, ignoreReductionWAW, eraseOutsideVarDef);
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
       << dep.later() << " in " << dep.later_.cursor_.node();
    os << " after ";
    os << (dep.earlier()->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << dep.earlier() << " in " << dep.earlier_.cursor_.node();
    bool first = true;
    for (auto &&[scope, dir] : dep.cond_) {
        os << (first ? " along " : " and ");
        first = false;
        os << scope.name_;
    }
    return std::regex_replace(os.str(), std::regex("\n"), "");
}

} // namespace ir

