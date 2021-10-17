#include <algorithm>
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
    if (op->noDeps_) {
        results_.emplace_back(op->id());
    }
}

void FindAccessPoint::visit(const VarDef &op) {
    defAxis_[op->name_] = cur_.size();
    defs_[op->name_] = op;
    Visitor::visit(op);
    defAxis_.erase(op->name_);
    defs_.erase(op->name_);
}

void FindAccessPoint::visit(const StmtSeq &op) {
    cur_.emplace_back(nullptr, makeIntConst(0),
                      makeIntConst(op->stmts_.size()));
    scope2coord_[op->id()] = cur_;
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        cur_.back().iter_ = makeIntConst(i);
        (*this)(op->stmts_[i]);
    }
    cur_.pop_back();
}

void FindAccessPoint::visit(const For &op) {
    cur_.emplace_back(makeVar(op->iter_), op->begin_, op->end_,
                      op->property_.parallel_);
    scope2coord_[op->id()] = cur_;
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const If &op) {
    (*this)(op->cond_);

    if (!op->elseCase_.isValid()) {
        auto oldCond = cond_;
        cond_ =
            oldCond.isValid() ? makeLAnd(oldCond, op->cond_) : (Expr)op->cond_;
        (*this)(op->thenCase_);
        cond_ = oldCond;
    } else {
        cur_.emplace_back(nullptr, makeIntConst(0), makeIntConst(2));
        scope2coord_[op->id()] = cur_;
        cur_.back().iter_ = makeIntConst(0);
        auto oldCond = cond_;
        cond_ =
            oldCond.isValid() ? makeLAnd(oldCond, op->cond_) : (Expr)op->cond_;
        (*this)(op->thenCase_);
        cur_.back().iter_ = makeIntConst(1);
        cond_ = oldCond.isValid() ? makeLAnd(oldCond, makeLNot(op->cond_))
                                  : makeLNot(op->cond_);
        (*this)(op->elseCase_);
        cond_ = oldCond;
        cur_.pop_back();
    }
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op,
           cursor(),
           defs_.at(op->var_),
           defs_.at(op->var_)->buffer_,
           defAxis_.at(op->var_),
           cur_,
           std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
           cond_};
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

std::string AnalyzeDeps::makeIterList(const std::vector<IterAxis> &list,
                                      int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret +=
                    genISLExpr_.normalizeId(list[i].iter_.as<VarNode>()->name_);
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

Ref<std::string> AnalyzeDeps::makeAccList(const std::vector<Expr> &list,
                                          RelaxMode relax,
                                          ExternalMap &externals) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genISLExpr_.gen(list[i]); linstr.isValid()) {
            ret += *linstr;
            unionTo(externals, genISLExpr_.externals(list[i]));
        } else if (relax == RelaxMode::Possible) {
            ret += genISLExpr_.normalizeId("free" + std::to_string(i));
        } else {
            return nullptr;
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return Ref<std::string>::make("[" + ret + "]");
}

Ref<std::string> AnalyzeDeps::makeRange(const std::vector<IterAxis> &point,
                                        RelaxMode relax,
                                        ExternalMap &externals) {
    std::vector<std::string> ineqs;
    for (size_t i = 0, iEnd = point.size(); i < iEnd; i++) {
        if (point[i].iter_->nodeType() == ASTNodeType::Var) {
            std::string ineq =
                genISLExpr_.normalizeId(point[i].iter_.as<VarNode>()->name_);
            bool bounded = true;
            if (auto linstr = genISLExpr_.gen(point[i].begin_);
                linstr.isValid()) {
                ineq = *linstr + " <= " + ineq;
                unionTo(externals, genISLExpr_.externals(point[i].begin_));
            } else {
                ineq = "free_lo <= " + ineq;
                bounded = false;
            }
            if (auto linstr = genISLExpr_.gen(point[i].end_);
                linstr.isValid()) {
                ineq = ineq + " < " + *linstr;
                unionTo(externals, genISLExpr_.externals(point[i].end_));
            } else {
                ineq = ineq + " < free_hi";
                bounded = false;
            }
            if (!bounded && relax == RelaxMode::Necessary) {
                return nullptr;
            }
            ineqs.emplace_back(std::move(ineq));
        }
    }
    std::string ret;
    for (size_t i = 0, iEnd = ineqs.size(); i < iEnd; i++) {
        ret += i == 0 ? "" : " and ";
        ret += ineqs[i];
    }
    return Ref<std::string>::make(std::move(ret));
}

Ref<std::string> AnalyzeDeps::makeCond(const Expr &expr, RelaxMode relax,
                                       ExternalMap &externals) {
    if (expr.isValid()) {
        if (auto str = genISLExpr_.gen(expr); str.isValid()) {
            unionTo(externals, genISLExpr_.externals(expr));
            return Ref<std::string>::make(*str);
        } else if (relax == RelaxMode::Necessary) {
            return nullptr;
        }
    }
    return Ref<std::string>::make("");
}

ISLMap AnalyzeDeps::makeAccMap(const AccessPoint &p, int iterDim, int accDim,
                               RelaxMode relax, const std::string &extSuffix,
                               ExternalMap &externals) {
    auto ret = makeIterList(p.iter_, iterDim) + " -> ";
    if (auto str = makeAccList(p.access_, relax, externals); str.isValid()) {
        ret += *str;
    } else {
        return emptyMap(spaceAlloc(isl_, 0, iterDim, accDim));
    }
    std::string cond;
    if (auto str = makeRange(p.iter_, relax, externals); str.isValid()) {
        cond += *str;
    } else {
        return emptyMap(spaceAlloc(isl_, 0, iterDim, accDim));
    }
    if (auto str = makeCond(p.cond_, relax, externals); str.isValid()) {
        cond += (cond.empty() || str->empty() ? "" : " and ") + *str;
    } else {
        return emptyMap(spaceAlloc(isl_, 0, iterDim, accDim));
    }
    if (!cond.empty()) {
        ret += ": " + cond;
    }
    std::string ext = "free_lo, free_hi";
    for (auto &&[hash, item] : externals) {
        ext += ", " + item.second;
    }
    ret = "[" + ext + "] -> {" + ret + "}";
    ret = replaceAll(ret, "!!placeholder!!", extSuffix);
    return ISLMap(isl_, ret);
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
AnalyzeDeps::makeEqForBothOps(const std::vector<std::pair<int, int>> &coord,
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
    return ISLMap(isl_, os.str());
}

ISLMap AnalyzeDeps::makeIneqBetweenOps(DepDirection mode, int iterId,
                                       int iterDim) const {
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
    return ISLMap(isl_, "{" + makeNdList("d", iterDim) + " -> " +
                            makeNdList("d_", iterDim) + ": d_" + idStr + " " +
                            ineq + " d" + idStr + "}");
}

ISLMap AnalyzeDeps::makeConstraintOfSingleLoop(const std::string &loop,
                                               DepDirection mode, int iterDim) {
    auto &&coord = scope2coord_.at(loop);
    int iterId = coord.size() - 1;
    if (iterId >= iterDim) {
        return emptyMap(spaceAlloc(isl_, 0, iterDim, iterDim));
    }

    auto ret = universeMap(spaceAlloc(isl_, 0, iterDim, iterDim));

    // Position in the outer StmtSeq nodes
    std::vector<std::pair<int, int>> pos;
    for (int i = 0; i < iterId; i++) {
        if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
            pos.emplace_back(i, coord[i].iter_.as<IntConstNode>()->val_);
        }
    }
    if (!pos.empty()) {
        ret = intersect(std::move(ret), makeEqForBothOps(pos, iterDim));
    }

    return intersect(std::move(ret), makeIneqBetweenOps(mode, iterId, iterDim));
}

ISLMap AnalyzeDeps::makeConstraintOfParallelScope(
    const std::string &parallel, DepDirection mode, int iterDim,
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
        return emptyMap(spaceAlloc(isl_, 0, iterDim, iterDim));
    }
    if (otherDim == -1 || pointDim == -1) {
        return universeMap(spaceAlloc(isl_, 0, iterDim, iterDim));
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
    return ISLMap(isl_, "{" + makeNdList("d", iterDim) + " -> " +
                            makeNdList("d_", iterDim) + ": d_" +
                            std::to_string(otherDim) + " " + ineq + " d" +
                            std::to_string(pointDim) + "}");
}

ISLMap AnalyzeDeps::makeExternalEq(int iterDim, const std::string &ext1,
                                   const std::string &ext2) {
    std::string mapping =
        makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim);
    return ISLMap(isl_, "[" + ext1 + ", " + ext2 + "] -> {" + mapping + ": " +
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

ISLMap AnalyzeDeps::makeSerialToAll(int iterDim, int serialIterDim,
                                    const std::vector<IterAxis> &point) const {
    std::string ret =
        makeNdList("d", serialIterDim) + " -> " + makeNdList("d_", iterDim);
    bool first = true;
    int j = 0;
    for (int i = 0; i < iterDim; i++) {
        if (i < (int)point.size()) {
            if (point[i].parallel_.empty()) {
                ret += first ? ": " : " and ";
                ret += "d" + std::to_string(j++) + " = d_" + std::to_string(i);
                first = false;
            }
        } else {
            ret += first ? ": " : " and ";
            ret += "d_" + std::to_string(i) + " = 0";
            first = false;
        }
    }
    while (j < serialIterDim) {
        ret += first ? ": " : " and ";
        ret += "d" + std::to_string(j++) + " = 0";
        first = false;
    }
    return ISLMap(isl_, "{" + ret + "}");
}

int AnalyzeDeps::countSerial(const std::vector<IterAxis> &point) {
    int ret = 0;
    for (auto &&item : point) {
        if (item.parallel_.empty()) {
            ret++;
        }
    }
    return ret;
}

ISLMap AnalyzeDeps::makeEraseVarDefConstraint(const Ref<AccessPoint> &point,
                                              int iterDim) {
    ISLMap ret = universeMap(spaceAlloc(isl_, 0, iterDim, iterDim));
    if (eraseOutsideVarDef_) {
        for (int i = 0; i < point->defAxis_; i++) {
            ret = intersect(std::move(ret),
                            makeIneqBetweenOps(DepDirection::Same, i, iterDim));
        }
    }
    return ret;
}

ISLMap AnalyzeDeps::makeNoDepsConstraint(int iterDim) {
    ISLMap ret = universeMap(spaceAlloc(isl_, 0, iterDim, iterDim));
    for (auto &&noDepsLoop : noDepsList_) {
        auto noDep = makeConstraintOfSingleLoop(
            noDepsLoop, DepDirection::Different, iterDim);
        ret = subtract(std::move(ret), std::move(noDep));
    }
    return ret;
}

ISLMap AnalyzeDeps::makeExternalVarConstraint(const Ref<AccessPoint> &point,
                                              const Ref<AccessPoint> &other,
                                              const ExternalMap &pExternals,
                                              const ExternalMap &oExternals,
                                              int iterDim) {
    ISLMap ret = universeMap(spaceAlloc(isl_, 0, iterDim, iterDim));
    auto opExternals = pExternals;
    unionTo(opExternals, oExternals);
    // We only have to add constraint for common loops of both accesses
    auto common = lca(point->cursor_, other->cursor_);

    // If all of the loops are variant, we don't have to make the constarint at
    // all. This will save time for ISL
    for (auto c = common;; c = c.outer()) {
        if (c.nodeType() == ASTNodeType::For) {
            for (auto &&[hash, item] : opExternals) {
                if (isVariant(variantExpr_, item.first, c.id())) {
                    goto found;
                }
            }
            goto do_compute_constarint;
        found:;
        }
        if (!c.hasOuter()) {
            break;
        }
    }
    return ret;

    // Compute the constraint
do_compute_constarint:
    for (auto c = common;; c = c.outer()) {
        if (c.nodeType() == ASTNodeType::For) {
            for (auto &&[hash, item] : opExternals) {
                if (isVariant(variantExpr_, item.first, c.id())) {
                    // Since idx[i] must be inside loop i, we only have
                    // to call makeIneqBetweenOps, but no need to call
                    // makeConstraintOfSingleLoop
                    auto diffIter = makeIneqBetweenOps(
                        DepDirection::Different,
                        scope2coord_.at(c.id()).size() - 1, iterDim);
                    auto sameExt = makeExternalEq(
                        iterDim,
                        replaceAll(item.second, "!!placeholder!!", "__ext_p"),
                        replaceAll(item.second, "!!placeholder!!", "__ext_o"));
                    auto require = uni(std::move(diffIter), std::move(sameExt));
                    ret = intersect(std::move(ret), std::move(require));
                }
            }
        }
        if (!c.hasOuter()) {
            break;
        }
    }
    return ret;
}

void AnalyzeDeps::checkDep(const Ref<AccessPoint> &point,
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
    int serialIterDim = countSerial(point->iter_);
    std::vector<bool> filteredIn(otherList.size());
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        filteredIn[i] = filter_ == nullptr || filter_(*point, *other);
        if (!filteredIn[i]) {
            continue;
        }

        iterDim = std::max<int>(iterDim, other->iter_.size());
        serialIterDim = std::max(serialIterDim, countSerial(other->iter_));
        ASSERT((int)other->access_.size() == accDim);
    }
    if (std::find(filteredIn.begin(), filteredIn.end(), true) ==
        filteredIn.end()) {
        return;
    }

    // lex_ge in serialDepAll AND ne in depAll
    ISLMap serialLexGE = lexGE(spaceSetAlloc(isl_, 0, serialIterDim));
    ISLMap allEQ = identity(spaceAlloc(isl_, 0, iterDim, iterDim));
    ISLMap eraseVarDefConstraint = makeEraseVarDefConstraint(point, iterDim);
    ISLMap noDepsConstraint = makeNoDepsConstraint(iterDim);

    ExternalMap pExternals;
    ISLMap pmap =
        makeAccMap(*point, iterDim, accDim, pRelax, "__ext_p", pExternals);
    if (pmap.empty()) {
        return;
    }
    ISLMap ps2a = makeSerialToAll(iterDim, serialIterDim, point->iter_);
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
            makeAccMap(*other, iterDim, accDim, oRelax, "__ext_o", oExternals);
        if (omap.empty()) {
            filteredIn[i] = false;
            continue;
        }
        ISLMap os2a = makeSerialToAll(iterDim, serialIterDim, other->iter_);
        ISLMap oa2s = reverse(os2a);
        ISLSet oIter = domain(omap);

        ISLMap depAll =
            subtract(applyRange(pmap, reverse(std::move(omap))), allEQ);

        depAll = intersect(std::move(depAll), eraseVarDefConstraint);
        depAll = intersect(std::move(depAll), noDepsConstraint);
        depAll = intersect(std::move(depAll),
                           makeExternalVarConstraint(point, other, pExternals,
                                                     oExternals, iterDim));

        ISLMap psDepAll = applyRange(depAll, std::move(oa2s));
        psDepAllUnion = psDepAllUnion.isValid()
                            ? uni(std::move(psDepAllUnion), std::move(psDepAll))
                            : std::move(psDepAll);

        os2aList[i] = std::move(os2a);
        oIterList[i] = std::move(oIter);
        depAllList[i] = std::move(depAll);
    }

    ISLMap ssDepAll = applyRange(std::move(ps2a), psDepAllUnion);
    ISLMap ssDep = intersect(std::move(ssDepAll), std::move(serialLexGE));
    ISLMap psDep = intersect(applyRange(std::move(pa2s), std::move(ssDep)),
                             std::move(psDepAllUnion));
    ISLMap psNearest = lexmax(std::move(psDep));

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
                    require = makeConstraintOfSingleLoop(nodeOrParallel.name_,
                                                         dir, iterDim);
                } else {
                    require = makeConstraintOfParallelScope(
                        nodeOrParallel.name_, dir, iterDim, point, other);
                }
                res = intersect(std::move(res), std::move(require));
                if (res.empty()) {
                    fail = true;
                    break;
                }
            }
            if (!fail) {
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
    if (depType_ & DEP_RAW) {
        auto &&point = points_.at(op);
        auto &&defId = defId_.at(op->var_);
        if (writes_.count(defId)) {
            checkDep(point, writes_.at(defId));
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

    FindAccessPoint accFinder;
    accFinder(op);
    FindAllNoDeps noDepsFinder;
    noDepsFinder(op);
    auto variantExpr = findLoopVariance(op).first;
    AnalyzeDeps analyzer(
        accFinder.points(), accFinder.reads(), accFinder.writes(),
        accFinder.scope2coord(), noDepsFinder.results(), variantExpr, cond,
        found, mode, depType, filter, ignoreReductionWAW, eraseOutsideVarDef);
    analyzer(op);
}

} // namespace ir

