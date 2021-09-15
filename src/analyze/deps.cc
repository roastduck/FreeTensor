#include <algorithm>
#include <sstream>

#include <analyze/deps.h>
#include <except.h>
#include <mutator.h>
#include <pass/simplify.h>

namespace ir {

static bool checkInnerScopeCrossThreads(const std::string &parallel) {
    if (parallel == "threadIdx.x" || parallel == "threadIdx.y" ||
        parallel == "threadIdx.z") {
        return true;
    }
    if (parallel == "blockIdx.x" || parallel == "blockIdx.y" ||
        parallel == "blockIdx.z") {
        return true;
    }
    return false;
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
                      !op->parallel_.empty(),
                      checkInnerScopeCrossThreads(op->parallel_));
    scope2coord_[op->id()] = cur_;
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const If &op) {
    (*this)(op->cond_);

    auto oldCond = cond_;
    cond_ = oldCond.isValid() ? makeLAnd(oldCond, op->cond_) : (Expr)op->cond_;
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        cond_ = oldCond.isValid() ? makeLAnd(oldCond, makeLNot(op->cond_))
                                  : makeLNot(op->cond_);
        (*this)(op->elseCase_);
    }
    cond_ = oldCond;
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op,
           cursor(),
           defs_.at(op->var_)->id(),
           defs_.at(op->var_)->buffer_,
           defAxis_.at(op->var_),
           cur_,
           std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
           cond_};
    points_.emplace(op, ap);
    reads_[defs_.at(op->var_)->id()].emplace_back(ap);
}

void GenISLExpr::reset() {
    visited_.clear();
    externals_.clear();
}

std::string GenISLExpr::normalizeId(const std::string &old) {
    if (idCache_.count(old)) {
        return idCache_.at(old);
    }
    std::string ret = old;
    for (char &c : ret) {
        if (!isalnum(c) && c != '_') {
            c = '_';
        }
    }
    while (idFlag_.count(ret)) {
        ret += "_";
    }
    idFlag_.insert(ret);
    return idCache_[old] = ret;
}

void GenISLExpr::visitExpr(const Expr &op,
                           const std::function<void(const Expr &)> &visitNode) {
    if (!visited_.count(op)) {
        Visitor::visitExpr(op, visitNode);
        visited_.insert(op);
    }
}

void GenISLExpr::visit(const Var &op) { results_[op] = normalizeId(op->name_); }

void GenISLExpr::visit(const IntConst &op) {
    results_[op] = std::to_string(op->val_);
    constants_[op] = op->val_;
}

void GenISLExpr::visit(const Load &op) {
    for (auto &&idx : op->indices_) {
        if (!constants_.count(idx)) {
            return;
        }
    }
    std::string str = op->var_ + ":";
    for (auto &&idx : op->indices_) {
        str += results_.at(idx) + ",";
    }
    externals_.insert(results_[op] = normalizeId(str));
}

void GenISLExpr::visit(const Add &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " + " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) +
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Sub &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " - " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) -
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Mul &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        if (constants_.count(op->lhs_) || constants_.count(op->rhs_)) {
            results_[op] = "(" + results_.at(op->lhs_) + " * " +
                           results_.at(op->rhs_) + ")";
        }
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) *
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const LAnd &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " and " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const LOr &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " or " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const LNot &op) {
    Visitor::visit(op);
    if (results_.count(op->expr_)) {
        results_[op] = "not " + results_.at(op->expr_);
    }
}

void GenISLExpr::visit(const LT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " < " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const LE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " <= " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const GT &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " > " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const GE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " >= " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const EQ &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " = " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const NE &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] = results_.at(op->lhs_) + " != " + results_.at(op->rhs_);
    }
}

void GenISLExpr::visit(const FloorDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] = "floor(" + results_.at(op->lhs_) + " / " +
                       results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    floorDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const CeilDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] = "ceil(" + results_.at(op->lhs_) + " / " +
                       results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    ceilDiv(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const Mod &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && constants_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " % " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] =
                std::to_string(constants_[op] = constants_.at(op->lhs_) %
                                                constants_.at(op->rhs_));
        }
    }
}

void GenISLExpr::visit(const Min &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "min(" + results_.at(op->lhs_) + ", " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    std::min(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

void GenISLExpr::visit(const Max &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "max(" + results_.at(op->lhs_) + ", " + results_.at(op->rhs_) + ")";
        if (constants_.count(op->lhs_) && constants_.count(op->rhs_)) {
            results_[op] = std::to_string(
                constants_[op] =
                    std::max(constants_.at(op->lhs_), constants_.at(op->rhs_)));
        }
    }
}

Ref<std::string> GenISLExpr::gen(const Expr &op) {
    (*this)(op);
    if (results_.count(op)) {
        return Ref<std::string>::make(results_.at(op));
    }
    return nullptr;
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
                                          RelaxMode relax) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genISLExpr_.gen(list[i]); linstr.isValid()) {
            ret += *linstr;
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
                                        RelaxMode relax) {
    std::vector<std::string> ineqs;
    for (size_t i = 0, iEnd = point.size(); i < iEnd; i++) {
        if (point[i].iter_->nodeType() == ASTNodeType::Var) {
            std::string ineq =
                genISLExpr_.normalizeId(point[i].iter_.as<VarNode>()->name_);
            bool bounded = true;
            if (auto linstr = genISLExpr_.gen(point[i].begin_);
                linstr.isValid()) {
                ineq = *linstr + " <= " + ineq;
            } else {
                ineq = "free_lo <= " + ineq;
                bounded = false;
            }
            if (auto linstr = genISLExpr_.gen(point[i].end_);
                linstr.isValid()) {
                ineq = ineq + " < " + *linstr;
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

Ref<std::string> AnalyzeDeps::makeCond(const Expr &expr, RelaxMode relax) {
    if (expr.isValid()) {
        if (auto str = genISLExpr_.gen(expr); str.isValid()) {
            return Ref<std::string>::make(*str);
        } else if (relax == RelaxMode::Necessary) {
            return nullptr;
        }
    }
    return Ref<std::string>::make("");
}

Ref<std::string> AnalyzeDeps::makeAccMap(const AccessPoint &p, int iterDim,
                                         int accDim, RelaxMode relax) {
    auto ret = makeIterList(p.iter_, iterDim) + " -> ";
    if (auto str = makeAccList(p.access_, relax); str.isValid()) {
        ret += *str;
    } else {
        return nullptr;
    }
    std::string cond;
    if (auto str = makeRange(p.iter_, relax); str.isValid()) {
        cond += *str;
    } else {
        return nullptr;
    }
    if (auto str = makeCond(p.cond_, relax); str.isValid()) {
        cond += (cond.empty() || str->empty() ? "" : " and ") + *str;
    } else {
        return nullptr;
    }
    if (!cond.empty()) {
        ret += ": " + cond;
    }
    std::string ext = "free_lo, free_hi";
    for (auto &&item : genISLExpr_.externals()) {
        ext += ", " + item;
    }
    return Ref<std::string>::make("[" + ext + "] -> {" + ret + "}");
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

std::string
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
    return os.str();
}

std::string AnalyzeDeps::makeIneqBetweenOps(DepDirection mode, int iterId,
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
    return "{" + makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim) +
           ": d_" + idStr + " " + ineq + " d" + idStr + "}";
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

std::string
AnalyzeDeps::makeSerialToAll(int iterDim, int serialIterDim,
                             const std::vector<IterAxis> &point) const {
    std::string ret =
        makeNdList("d", serialIterDim) + " -> " + makeNdList("d_", iterDim);
    bool first = true;
    int j = 0;
    for (int i = 0; i < iterDim; i++) {
        if (i < (int)point.size()) {
            if (!point[i].parallel_) {
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
    return "{" + ret + "}";
}

int AnalyzeDeps::countSerial(const std::vector<IterAxis> &point) {
    int ret = 0;
    for (auto &&item : point) {
        if (!item.parallel_) {
            ret++;
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

    // We call genISLExpr_ twice. The first time is to warm up the external
    // list, because all the maps shall have an identical external list

    // 1st time
    genISLExpr_.reset();
    if (!makeAccMap(*point, iterDim, accDim, pRelax).isValid()) {
        return;
    }
    for (size_t i = 0, n = otherList.size(); i < n; i++) {
        auto &&other = otherList[i];
        if (!filteredIn[i]) {
            continue;
        }

        filteredIn[i] = makeAccMap(*other, iterDim, accDim, oRelax).isValid();
    }
    if (std::find(filteredIn.begin(), filteredIn.end(), true) ==
        filteredIn.end()) {
        return;
    }

    // 2nd time
    ISLMap pmap(isl_, *makeAccMap(*point, iterDim, accDim, pRelax));
    ISLMap ps2a(isl_, makeSerialToAll(iterDim, serialIterDim, point->iter_));
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

        ISLMap omap(isl_, *makeAccMap(*other, iterDim, accDim, oRelax));
        ISLMap os2a(isl_,
                    makeSerialToAll(iterDim, serialIterDim, other->iter_));
        ISLMap oa2s = reverse(os2a);
        ISLSet oIter = domain(omap);

        ISLMap depAll =
            subtract(applyRange(pmap, reverse(std::move(omap))), allEQ);
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

        bool pFullyKilled = pIter == domain(nearest);
        bool oFullyKilled = oIter == range(nearest);

        if (eraseOutsideVarDef_) {
            for (int i = 0; i < point->defAxis_; i++) {
                if (point->iter_[i].iter_->nodeType() == ASTNodeType::Var &&
                    !point->iter_[i]
                         .innerScopeCrossThreads_) { // is a loop and not
                                                     // crosing threads
                    ISLMap restriction(
                        isl_,
                        makeIneqBetweenOps(DepDirection::Same, i, iterDim));
                    nearest =
                        intersect(std::move(nearest), std::move(restriction));
                }
            }
        }

        if (nearest.empty()) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillEarlier ||
             mode_ == FindDepsMode::KillBoth) &&
            !oFullyKilled) {
            continue;
        }
        if ((mode_ == FindDepsMode::KillLater ||
             mode_ == FindDepsMode::KillBoth) &&
            !pFullyKilled) {
            continue;
        }

        for (auto &&item : cond_) {
            bool found = true;
            for (auto &&subitem : item) {
                auto &&coord = scope2coord_.at(subitem.first);
                int iterId = coord.size() - 1;
                DepDirection mode = subitem.second;
                if (iterId >= iterDim) {
                    found = false;
                    break;
                }
                ISLMap res = nearest;

                // Position in the outer StmtSeq nodes
                std::vector<std::pair<int, int>> pos;
                for (int i = 0; i < iterId; i++) {
                    if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
                        pos.emplace_back(
                            i, coord[i].iter_.as<IntConstNode>()->val_);
                    }
                }
                if (!pos.empty()) {
                    ISLMap require(isl_, makeEqForBothOps(pos, iterDim));
                    res = intersect(std::move(res), std::move(require));
                }

                ISLMap require(isl_, makeIneqBetweenOps(mode, iterId, iterDim));
                res = intersect(std::move(res), std::move(require));
                found &= !res.empty();
            }
            if (found) {
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

void findDeps(
    const Stmt &op,
    const std::vector<std::vector<std::pair<std::string, DepDirection>>> &cond,
    const FindDepsCallback &found, FindDepsMode mode, DepType depType,
    const FindDepsFilter &filter, bool ignoreReductionWAW,
    bool eraseOutsideVarDef) {
    if (cond.empty()) {
        return;
    }

    FindAccessPoint finder;
    finder(op);
    AnalyzeDeps analyzer(finder.points(), finder.reads(), finder.writes(),
                         finder.scope2coord(), cond, found, mode, depType,
                         filter, ignoreReductionWAW, eraseOutsideVarDef);
    analyzer(op);
}

} // namespace ir

