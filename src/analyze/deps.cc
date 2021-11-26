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
    conds_.emplace_back(makeGE(iter, op->begin_));
    conds_.emplace_back(makeLT(iter, op->end_));
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
    conds_.pop_back();
    conds_.pop_back();
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
    points_.emplace(op, ap);
    reads_[defs_.at(op->var_)->id()].emplace_back(ap);
}

void GenISLExprDeps::visitExpr(
    const Expr &op, const std::function<void(const Expr &)> &visitNode) {
    auto oldParent = parent_;
    parent_ = op;
    GenISLExpr::visitExpr(op, visitNode);
    parent_ = oldParent;
    if (parent_.isValid()) {
        unionTo(externals_[parent_], externals_[op]);
    }
}

void GenISLExprDeps::visit(const Load &op) {
    getHash_(op);
    auto h = getHash_.hash().at(op);
    auto str = normalizeId("ext" + std::to_string(h)) + "!!placeholder!!";
    externals_[op][h] = std::make_pair(op, str);
    results_[op] = str;
}

std::string AnalyzeDeps::makeIterList(GenISLExprDeps &genISLExpr,
                                      const std::vector<IterAxis> &list,
                                      int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret +=
                    genISLExpr.normalizeId(list[i].iter_.as<VarNode>()->name_);
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

Ref<std::string> AnalyzeDeps::makeAccList(GenISLExprDeps &genISLExpr,
                                          const std::vector<Expr> &list,
                                          RelaxMode relax,
                                          ExternalMap &externals) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genISLExpr.gen(list[i]); linstr.isValid()) {
            ret += *linstr;
            unionTo(externals, genISLExpr.externals(list[i]));
        } else if (relax == RelaxMode::Possible) {
            ret += genISLExpr.normalizeId("free" + std::to_string(i));
        } else {
            return nullptr;
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return Ref<std::string>::make("[" + ret + "]");
}

Ref<std::string> AnalyzeDeps::makeCond(GenISLExprDeps &genISLExpr,
                                       const std::vector<Expr> &conds,
                                       RelaxMode relax,
                                       ExternalMap &externals) {
    std::string ret;
    for (auto &&cond : conds) {
        if (auto str = genISLExpr.gen(cond); str.isValid()) {
            unionTo(externals, genISLExpr.externals(cond));
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

ISLMap AnalyzeDeps::makeAccMap(ISLCtx &isl, GenISLExprDeps &genISLExpr,
                               const AccessPoint &p, int iterDim, int accDim,
                               RelaxMode relax, const std::string &extSuffix,
                               ExternalMap &externals) {
    auto ret = makeIterList(genISLExpr, p.iter_, iterDim) + " -> ";
    if (auto str = makeAccList(genISLExpr, p.access_, relax, externals);
        str.isValid()) {
        ret += *str;
    } else {
        return emptyMap(spaceAlloc(isl, 0, iterDim, accDim));
    }
    std::string cond;
    if (auto str = makeCond(genISLExpr, p.conds_, relax, externals);
        str.isValid()) {
        cond += (cond.empty() || str->empty() ? "" : " and ") + *str;
    } else {
        return emptyMap(spaceAlloc(isl, 0, iterDim, accDim));
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
    return ISLMap(isl, ret);
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

ISLMap
AnalyzeDeps::makeEqForBothOps(ISLCtx &isl,
                              const std::vector<std::pair<int, int>> &coord,
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
    return ISLMap(isl, os.str());
}

ISLMap AnalyzeDeps::makeIneqBetweenOps(ISLCtx &isl, DepDirection mode,
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
    return ISLMap(isl, "{" + makeNdList("d", iterDim) + " -> " +
                           makeNdList("d_", iterDim) + ": d_" + idStr + " " +
                           ineq + " d" + idStr + "}");
}

ISLMap AnalyzeDeps::makeConstraintOfSingleLoop(ISLCtx &isl,
                                               const std::string &loop,
                                               DepDirection mode, int iterDim) {
    auto &&coord = scope2coord_.at(loop);
    int iterId = coord.size() - 1;
    if (iterId >= iterDim) {
        return emptyMap(spaceAlloc(isl, 0, iterDim, iterDim));
    }

    auto ret = universeMap(spaceAlloc(isl, 0, iterDim, iterDim));

    // Position in the outer StmtSeq nodes
    std::vector<std::pair<int, int>> pos;
    for (int i = 0; i < iterId; i++) {
        if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
            pos.emplace_back(i, coord[i].iter_.as<IntConstNode>()->val_);
        }
    }
    if (!pos.empty()) {
        ret = intersect(std::move(ret), makeEqForBothOps(isl, pos, iterDim));
    }

    return intersect(std::move(ret),
                     makeIneqBetweenOps(isl, mode, iterId, iterDim));
}

ISLMap AnalyzeDeps::makeConstraintOfParallelScope(
    ISLCtx &isl, const std::string &parallel, DepDirection mode, int iterDim,
    const Ref<AccessPoint> &point, const Ref<AccessPoint> &other) {
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
        return emptyMap(spaceAlloc(isl, 0, iterDim, iterDim));
    }
    if (otherDim == -1 || pointDim == -1) {
        return universeMap(spaceAlloc(isl, 0, iterDim, iterDim));
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
    return ISLMap(isl, "{" + makeNdList("d", iterDim) + " -> " +
                           makeNdList("d_", iterDim) + ": d_" +
                           std::to_string(otherDim) + " " + ineq + " d" +
                           std::to_string(pointDim) + "}");
}

ISLMap AnalyzeDeps::makeExternalEq(ISLCtx &isl, int iterDim,
                                   const std::string &ext1,
                                   const std::string &ext2) {
    std::string mapping =
        makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim);
    return ISLMap(isl, "[" + ext1 + ", " + ext2 + "] -> {" + mapping + ": " +
                           ext1 + " = " + ext2 + "}");
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

ISLMap AnalyzeDeps::makeSerialToAll(ISLCtx &isl, int iterDim,
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
    return ISLMap(isl, "{" + from + " -> " + to + "}");
}

ISLMap AnalyzeDeps::makeEraseVarDefConstraint(ISLCtx &isl,
                                              const Ref<AccessPoint> &point,
                                              int iterDim) {
    ISLMap ret = universeMap(spaceAlloc(isl, 0, iterDim, iterDim));
    if (eraseOutsideVarDef_) {
        for (int i = 0; i < point->defAxis_; i++) {
            ret = intersect(
                std::move(ret),
                makeIneqBetweenOps(isl, DepDirection::Same, i, iterDim));
        }
    }
    return ret;
}

ISLMap AnalyzeDeps::makeNoDepsConstraint(ISLCtx &isl, const std::string &var,
                                         int iterDim) {
    ISLMap ret = universeMap(spaceAlloc(isl, 0, iterDim, iterDim));
    if (noDepsLists_.count(var)) {
        for (auto &&noDepsLoop : noDepsLists_.at(var)) {
            auto noDep = makeConstraintOfSingleLoop(
                isl, noDepsLoop, DepDirection::Different, iterDim);
            ret = subtract(std::move(ret), std::move(noDep));
        }
    }
    return ret;
}

ISLMap AnalyzeDeps::makeExternalVarConstraint(
    ISLCtx &isl, const Ref<AccessPoint> &point, const Ref<AccessPoint> &other,
    const ExternalMap &pExternals, const ExternalMap &oExternals, int iterDim,
    const std::string &extSuffixP, const std::string &extSuffixO) {
    ISLMap ret = universeMap(spaceAlloc(isl, 0, iterDim, iterDim));
    // We only have to add constraint for common loops of both accesses
    auto common = lca(point->cursor_, other->cursor_);

    for (auto &&[hash, item] : intersect(pExternals, oExternals)) {
        // If all of the loops are variant, we don't have to make the constraint
        // at all. This will save time for ISL
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
            isl, iterDim,
            replaceAll(item.second, "!!placeholder!!", extSuffixP),
            replaceAll(item.second, "!!placeholder!!", extSuffixO));
        for (auto c = common;; c = c.outer()) {
            if (c.nodeType() == ASTNodeType::For) {
                if (isVariant(variantExpr_, item.first, c.id())) {
                    // Since idx[i] must be inside loop i, we only have
                    // to call makeIneqBetweenOps, but no need to call
                    // makeConstraintOfSingleLoop
                    auto diffIter = makeIneqBetweenOps(
                        isl, DepDirection::Different,
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

ISLMap AnalyzeDeps::projectOutPrivateAxis(ISLCtx &isl, int iterDim, int since) {
    std::string from = makeNdList("d", iterDim);
    std::string to;
    for (int i = 0; i < iterDim; i++) {
        to += (i > 0 ? ", " : "") + (i < since ? "d" + std::to_string(i) : "0");
    }
    to = "[" + to + "]";
    return ISLMap(isl, "{" + from + " -> " + to + "}");
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
    const std::vector<Ref<AccessPoint>> &otherList) {
    tasks_.emplace_back([point, otherList, this]() {
        ISLCtx isl;
        GenISLExprDeps genISLExpr;
        checkDepLatestEarlierImpl(isl, genISLExpr, point, otherList);
    });
}

void AnalyzeDeps::checkDepEarliestLater(
    const std::vector<Ref<AccessPoint>> &pointList,
    const Ref<AccessPoint> &other) {
    tasks_.emplace_back([pointList, other, this]() {
        ISLCtx isl;
        GenISLExprDeps genISLExpr;
        checkDepEarliestLaterImpl(isl, genISLExpr, pointList, other);
    });
}

void AnalyzeDeps::checkDepLatestEarlierImpl(
    ISLCtx &isl, GenISLExprDeps &genISLExpr, const Ref<AccessPoint> &point,
    const std::vector<Ref<AccessPoint>> &otherList) {
    if (otherList.empty()) {
        return;
    }

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
    std::vector<bool> filteredIn(otherList.size());
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (filter_ == nullptr) {
            filteredIn[i] = true;
        } else {
            std::lock_guard<std::mutex> guard(lock_);
            filteredIn[i] = filter_(*point, *other);
        }
        if (!filteredIn[i]) {
            continue;
        }

        iterDim = std::max<int>(iterDim, other->iter_.size());
        ASSERT((int)other->access_.size() == accDim);
    }
    if (std::find(filteredIn.begin(), filteredIn.end(), true) ==
        filteredIn.end()) {
        return;
    }

    int pCommonDims = 0;
    std::vector<int> oCommonDims(otherList.size(), 0);
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (!filteredIn[i]) {
            continue;
        }

        int cpo = numCommonDims(point, other);
        pCommonDims = std::max(pCommonDims, cpo);
        oCommonDims[i] = std::max(oCommonDims[i], cpo);
        auto j = i + 1;
        while (j < n && !filteredIn[j]) {
            j++;
        }
        if (j < n) {
            int co1o2 = numCommonDims(other, otherList[j]);
            oCommonDims[i] = std::max(oCommonDims[i], co1o2);
            oCommonDims[j] = std::max(oCommonDims[j], co1o2);
        }
    }

    ISLMap allEQ = identity(spaceAlloc(isl, 0, iterDim, iterDim));
    ISLMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(isl, point, iterDim);
    ISLMap noDepsConstraint =
        makeNoDepsConstraint(isl, point->def_->name_, iterDim);

    ExternalMap pExternals;
    ISLMap pmap = makeAccMap(isl, genISLExpr, *point, iterDim, accDim, pRelax,
                             "__ext_p", pExternals);
    if (pmap.empty()) {
        return;
    }
    if (mode_ == FindDepsMode::Dep &&
        pCommonDims + 1 < (int)point->iter_.size()) {
        pmap =
            applyDomain(std::move(pmap),
                        projectOutPrivateAxis(isl, iterDim, pCommonDims + 1));
    }
    ISLMap ps2a = makeSerialToAll(isl, iterDim, point->iter_);
    ISLMap pa2s = reverse(ps2a);
    ISLSet pIter = domain(pmap);
    std::vector<ISLMap> os2aList(otherList.size()),
        depAllList(otherList.size());
    std::vector<ISLSet> oIterList(otherList.size());
    ISLMap psDepAllUnion;
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (!filteredIn[i]) {
            continue;
        }

        ExternalMap oExternals;
        ISLMap omap =
            makeAccMap(isl, genISLExpr, *other, iterDim, accDim, oRelax,
                       "__ext_o" + std::to_string(i), oExternals);
        if (omap.empty()) {
            filteredIn[i] = false;
            continue;
        }
        if (mode_ == FindDepsMode::Dep &&
            oCommonDims[i] + 1 < (int)other->iter_.size()) {
            omap = applyDomain(
                std::move(omap),
                projectOutPrivateAxis(isl, iterDim, oCommonDims[i] + 1));
        }
        ISLMap os2a = makeSerialToAll(isl, iterDim, other->iter_);
        ISLMap oa2s = reverse(os2a);
        ISLSet oIter = domain(omap);

        ISLMap depAll =
            subtract(applyRange(pmap, reverse(std::move(omap))), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll =
            intersect(std::move(depAll),
                      makeExternalVarConstraint(isl, point, other, pExternals,
                                                oExternals, iterDim, "__ext_p",
                                                "__ext_o" + std::to_string(i)));

        ISLMap psDepAll = applyRange(depAll, std::move(oa2s));
        psDepAllUnion = psDepAllUnion.isValid()
                            ? uni(std::move(psDepAllUnion), std::move(psDepAll))
                            : std::move(psDepAll);

        os2aList[i] = std::move(os2a);
        oIterList[i] = std::move(oIter);
        depAllList[i] = std::move(depAll);
    }
    if (!psDepAllUnion.isValid()) {
        return;
    }

    ISLMap serialLexGT = lexGT(spaceSetAlloc(isl, 0, iterDim));
    ISLMap serialEQ = identity(spaceAlloc(isl, 0, iterDim, iterDim));
    ISLMap ssDepAll = applyRange(std::move(ps2a), psDepAllUnion);
    ISLMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    ISLMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    ISLMap psDep = intersect(applyRange(pa2s, std::move(ssDep)), psDepAllUnion);
    ISLMap psSelf = intersect(applyRange(std::move(pa2s), std::move(ssSelf)),
                              std::move(psDepAllUnion));
    ISLMap psNearest = uni(lexmax(std::move(psDep)), std::move(psSelf));

    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (!filteredIn[i]) {
            continue;
        }

        auto &&os2a = os2aList[i];
        auto &&oIter = oIterList[i];
        auto &&depAll = depAllList[i];
        ISLMap nearest = intersect(applyRange(psNearest, std::move(os2a)),
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
            ISLMap res = nearest;
            bool fail = false;
            for (auto &&[nodeOrParallel, dir] : item) {
                ISLMap require;
                if (nodeOrParallel.isNode_) {
                    require = makeConstraintOfSingleLoop(
                        isl, nodeOrParallel.name_, dir, iterDim);
                } else {
                    require = makeConstraintOfParallelScope(
                        isl, nodeOrParallel.name_, dir, iterDim, point, other);
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
    ISLCtx &isl, GenISLExprDeps &genISLExpr,
    const std::vector<Ref<AccessPoint>> &pointList,
    const Ref<AccessPoint> &other) {
    if (pointList.empty()) {
        return;
    }

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
    std::vector<bool> filteredIn(pointList.size());
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        if (filter_ == nullptr) {
            filteredIn[i] = true;
        } else {
            std::lock_guard<std::mutex> guard(lock_);
            filteredIn[i] = filter_(*point, *other);
        }
        if (!filteredIn[i]) {
            continue;
        }

        iterDim = std::max<int>(iterDim, point->iter_.size());
        ASSERT((int)point->access_.size() == accDim);
    }
    if (std::find(filteredIn.begin(), filteredIn.end(), true) ==
        filteredIn.end()) {
        return;
    }

    int oCommonDims = 0;
    std::vector<int> pCommonDims(pointList.size(), 0);
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        if (!filteredIn[i]) {
            continue;
        }

        int cpo = numCommonDims(point, other);
        oCommonDims = std::max(oCommonDims, cpo);
        pCommonDims[i] = std::max(pCommonDims[i], cpo);
        auto j = i + 1;
        while (j < n && !filteredIn[j]) {
            j++;
        }
        if (j < n) {
            int cp1p2 = numCommonDims(point, pointList[j]);
            pCommonDims[i] = std::max(pCommonDims[i], cp1p2);
            pCommonDims[j] = std::max(pCommonDims[j], cp1p2);
        }
    }

    ISLMap allEQ = identity(spaceAlloc(isl, 0, iterDim, iterDim));
    ISLMap eraseVarDefConstraint =
        makeEraseVarDefConstraint(isl, other, iterDim);
    ISLMap noDepsConstraint =
        makeNoDepsConstraint(isl, other->def_->name_, iterDim);

    ExternalMap oExternals;
    ISLMap omap = makeAccMap(isl, genISLExpr, *other, iterDim, accDim, oRelax,
                             "__ext_o", oExternals);
    if (omap.empty()) {
        return;
    }
    if (mode_ == FindDepsMode::Dep &&
        oCommonDims + 1 < (int)other->iter_.size()) {
        omap =
            applyDomain(std::move(omap),
                        projectOutPrivateAxis(isl, iterDim, oCommonDims + 1));
    }
    ISLMap os2a = makeSerialToAll(isl, iterDim, other->iter_);
    ISLMap oa2s = reverse(os2a);
    ISLSet oIter = domain(omap);
    std::vector<ISLMap> ps2aList(pointList.size()),
        depAllList(pointList.size());
    std::vector<ISLSet> pIterList(pointList.size());
    ISLMap spDepAllUnion;
    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        if (!filteredIn[i]) {
            continue;
        }

        ExternalMap pExternals;
        ISLMap pmap =
            makeAccMap(isl, genISLExpr, *point, iterDim, accDim, pRelax,
                       "__ext_p" + std::to_string(i), pExternals);
        if (pmap.empty()) {
            filteredIn[i] = false;
            continue;
        }
        if (mode_ == FindDepsMode::Dep &&
            pCommonDims[i] + 1 < (int)point->iter_.size()) {
            pmap = applyDomain(
                std::move(pmap),
                projectOutPrivateAxis(isl, iterDim, pCommonDims[i] + 1));
        }
        ISLMap ps2a = makeSerialToAll(isl, iterDim, point->iter_);
        ISLMap pa2s = reverse(ps2a);
        ISLSet pIter = domain(pmap);

        ISLMap depAll =
            subtract(applyRange(std::move(pmap), reverse(omap)), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll =
            intersect(std::move(depAll),
                      makeExternalVarConstraint(
                          isl, point, other, pExternals, oExternals, iterDim,
                          "__ext_p" + std::to_string(i), "__ext_o"));

        ISLMap spDepAll = applyDomain(depAll, std::move(pa2s));
        spDepAllUnion = spDepAllUnion.isValid()
                            ? uni(std::move(spDepAllUnion), std::move(spDepAll))
                            : std::move(spDepAll);

        ps2aList[i] = std::move(ps2a);
        pIterList[i] = std::move(pIter);
        depAllList[i] = std::move(depAll);
    }
    if (!spDepAllUnion.isValid()) {
        return;
    }

    ISLMap serialLexGT = lexGT(spaceSetAlloc(isl, 0, iterDim));
    ISLMap serialEQ = identity(spaceAlloc(isl, 0, iterDim, iterDim));
    ISLMap ssDepAll = applyRange(spDepAllUnion, std::move(oa2s));
    ISLMap ssDep = intersect(ssDepAll, std::move(serialLexGT));
    ISLMap ssSelf = intersect(ssDepAll, std::move(serialEQ));
    ISLMap spDep = intersect(applyRange(std::move(ssDep), os2a), spDepAllUnion);
    ISLMap spSelf = intersect(applyRange(std::move(ssSelf), std::move(os2a)),
                              std::move(spDepAllUnion));
    ISLMap spNearest =
        uni(reverse(lexmin(reverse(std::move(spDep)))), std::move(spSelf));

    for (size_t i = 0, n = pointList.size(); i < n; i++) {
        auto &&point = pointList[i];
        if (!filteredIn[i]) {
            continue;
        }

        auto &&ps2a = ps2aList[i];
        auto &&pIter = pIterList[i];
        auto &&depAll = depAllList[i];
        ISLMap nearest = intersect(applyDomain(spNearest, std::move(ps2a)),
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
            ISLMap res = nearest;
            bool fail = false;
            for (auto &&[nodeOrParallel, dir] : item) {
                ISLMap require;
                if (nodeOrParallel.isNode_) {
                    require = makeConstraintOfSingleLoop(
                        isl, nodeOrParallel.name_, dir, iterDim);
                } else {
                    require = makeConstraintOfParallelScope(
                        isl, nodeOrParallel.name_, dir, iterDim, point, other);
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

void AnalyzeDeps::visit(const VarDef &op) {
    ASSERT(!defId_.count(op->name_));
    defId_[op->name_] = op->id();
    Visitor::visit(op);
    defId_.erase(op->name_);
}

void AnalyzeDeps::visit(const Load &op) {
    Visitor::visit(op);
    auto &&defId = defId_.at(op->var_);
    if (depType_ & DEP_RAW) {
        auto &&point = points_.at(op);
        if (writes_.count(defId)) {
            checkDepLatestEarlier(point, writes_.at(defId));
        }
    }
    if (depType_ & DEP_WAR) {
        auto &other = points_.at(op);
        if (writes_.count(defId)) {
            checkDepEarliestLater(writes_.at(defId), other);
        }
    }
}

void AnalyzeDeps::visit(const Store &op) {
    Visitor::visit(op);
    auto &&defId = defId_.at(op->var_);
    if (depType_ & DEP_WAW) {
        auto &&point = points_.at(op);
        if (writes_.count(defId)) {
            checkDepLatestEarlier(point, writes_.at(defId));
        }
    }
}

void AnalyzeDeps::visit(const ReduceTo &op) {
    Visitor::visit(op);
    auto &&defId = defId_.at(op->var_);

    if ((depType_ & DEP_RAW) || (depType_ & DEP_WAW)) {
        auto &&point = points_.at(op);
        if (writes_.count(defId)) {
            std::vector<Ref<AccessPoint>> others;
            for (auto &&item : writes_.at(defId)) {
                if (ignoreReductionWAW_ &&
                    item->op_->nodeType() == ASTNodeType::ReduceTo) {
                    continue;
                }
                others.emplace_back(item);
            }
            checkDepLatestEarlier(point, others);
        }
    }

    if (depType_ & DEP_WAR) {
        auto &&other = points_.at(op);
        if (writes_.count(defId)) {
            std::vector<Ref<AccessPoint>> points;
            for (auto &&item : writes_.at(defId)) {
                if (ignoreReductionWAW_ &&
                    item->op_->nodeType() == ASTNodeType::ReduceTo) {
                    continue;
                }
                points.emplace_back(item);
            }
            checkDepEarliestLater(points, other);
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
        accFinder.points(), accFinder.reads(), accFinder.writes(),
        accFinder.scope2coord(), noDepsFinder.results(), variantExpr, cond,
        found, mode, depType, filter, ignoreReductionWAW, eraseOutsideVarDef);
    analyzer(op);
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

