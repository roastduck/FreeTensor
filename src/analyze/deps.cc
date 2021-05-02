#include <algorithm>
#include <sstream>

#include <analyze/deps.h>
#include <except.h>
#include <mutator.h>
#include <pass/simplify.h>

namespace ir {

void FindAccessPoint::visit(const VarDef &op) {
    defAxis_[op->name_] = cur_.size();
    Visitor::visit(op);
    defAxis_.erase(op->name_);
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
    cur_.emplace_back(makeVar(op->iter_), op->begin_, op->end_);
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
           defAxis_.at(op->var_),
           cur_,
           std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
           cond_};
    points_.emplace(op, ap);
    reads_.emplace(op->var_, ap);
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
}

void GenISLExpr::visit(const Load &op) {
    for (auto &&idx : op->indices_) {
        if (idx->nodeType() != ASTNodeType::IntConst) {
            return;
        }
    }
    std::string str = op->var_ + ":";
    for (auto &&idx : op->indices_) {
        str += std::to_string(idx.as<IntConstNode>()->val_) + ",";
    }
    externals_.insert(results_[op] = normalizeId(str));
}

void GenISLExpr::visit(const Add &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " + " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const Sub &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " - " + results_.at(op->rhs_) + ")";
    }
}

void GenISLExpr::visit(const Mul &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) && results_.count(op->rhs_) &&
        (op->lhs_->nodeType() == ASTNodeType::IntConst ||
         op->rhs_->nodeType() == ASTNodeType::IntConst)) {
        results_[op] =
            "(" + results_.at(op->lhs_) + " * " + results_.at(op->rhs_) + ")";
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
    if (results_.count(op->lhs_) &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        results_[op] = "floor(" + results_.at(op->lhs_) + " / " +
                       std::to_string(op->rhs_.as<IntConstNode>()->val_) + ")";
    }
}

void GenISLExpr::visit(const CeilDiv &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        results_[op] = "ceil(" + results_.at(op->lhs_) + " / " +
                       std::to_string(op->rhs_.as<IntConstNode>()->val_) + ")";
    }
}

void GenISLExpr::visit(const Mod &op) {
    Visitor::visit(op);
    if (results_.count(op->lhs_) &&
        op->rhs_->nodeType() == ASTNodeType::IntConst) {
        results_[op] = "(" + results_.at(op->lhs_) + " % " +
                       std::to_string(op->rhs_.as<IntConstNode>()->val_) + ")";
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
                                      int eraseBefore, int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i].iter_->nodeType() == ASTNodeType::Var) {
                ret +=
                    genISLExpr_.normalizeId(list[i].iter_.as<VarNode>()->name_);
                if (i < eraseBefore) {
                    ret += " = 0";
                }
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
    auto ret = makeIterList(p.iter_, p.defAxis_, iterDim) + " -> ";
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
    case DepDirection::Same:
        ineq = "=";
        break;
    case DepDirection::Normal:
        ineq = "<";
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

void AnalyzeDeps::checkDep(const AccessPoint &point, const AccessPoint &other) {
    int iterDim = std::max(point.iter_.size(), other.iter_.size());
    int accDim = point.access_.size();
    ASSERT((int)other.access_.size() == accDim);

    auto pRelax = mode_ == FindDepsMode::Kill ? RelaxMode::Necessary
                                              : RelaxMode::Possible; // later
    auto oRelax = RelaxMode::Possible;                               // earlier

    // We call genISLExpr_ twice. The first time is to warm up the external
    // list, because all the maps shall have an identical external list

    // 1st time
    genISLExpr_.reset();
    if (!makeAccMap(point, iterDim, accDim, pRelax).isValid()) {
        return;
    }
    if (!makeAccMap(other, iterDim, accDim, oRelax).isValid()) {
        return;
    }

    // 2nd time
    isl_map *pmap = isl_map_read_from_str(
        isl_, makeAccMap(point, iterDim, accDim, pRelax)->c_str());
    isl_map *omap = isl_map_read_from_str(
        isl_, makeAccMap(other, iterDim, accDim, oRelax)->c_str());

    isl_set *domain = isl_set_read_from_str(
        isl_, ("{" + makeNdList("d", iterDim) + "}").c_str());
    isl_space *space = isl_set_get_space(domain);
    isl_set_free(domain);
    isl_map *pred = isl_map_lex_gt(space);

    isl_set *pIter = isl_map_domain(isl_map_copy(omap));
    isl_map *depall = isl_map_apply_range(pmap, isl_map_reverse(omap));
    isl_map *dep = isl_map_intersect(depall, pred);
    isl_map *nearest = isl_map_lexmax(dep);
    isl_set *pIterKilled = isl_map_range(isl_map_copy(nearest));
    bool fullyKilled = isl_set_is_equal(pIter, pIterKilled);
    isl_set_free(pIter);
    isl_set_free(pIterKilled);
    if (isl_map_is_empty(nearest) ||
        (mode_ == FindDepsMode::Kill && !fullyKilled)) {
        isl_map_free(nearest);
        return;
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
            isl_map *res = isl_map_copy(nearest);

            // Position in the outer StmtSeq nodes
            std::vector<std::pair<int, int>> pos;
            for (int i = 0; i < iterId; i++) {
                if (coord[i].iter_->nodeType() == ASTNodeType::IntConst) {
                    pos.emplace_back(i,
                                     coord[i].iter_.as<IntConstNode>()->val_);
                }
            }
            if (!pos.empty()) {
                isl_basic_map *require = isl_basic_map_read_from_str(
                    isl_, makeEqForBothOps(pos, iterDim).c_str());
                res = isl_map_intersect(res, isl_map_from_basic_map(require));
            }

            isl_basic_map *require = isl_basic_map_read_from_str(
                isl_, makeIneqBetweenOps(mode, iterId, iterDim).c_str());
            res = isl_map_intersect(res, isl_map_from_basic_map(require));
            found &= !isl_map_is_empty(res);
            isl_map_free(res);
        }
        if (found) {
            try {
                found_(Dependency{item, getVar(point.op_), point, other});
            } catch (...) {
                isl_map_free(nearest);
                throw;
            }
        }
    }
    isl_map_free(nearest);
}

void AnalyzeDeps::visit(const Load &op) {
    Visitor::visit(op);
    if (depType_ & DEP_RAW) {
        auto &&point = points_.at(op);
        auto range = writes_.equal_range(op->var_);
        for (auto i = range.first; i != range.second; i++) {
            if (filter_ != nullptr && !filter_(*point, *(i->second))) {
                continue;
            }
            checkDep(*point, *(i->second));
        }
    }
}

void findDeps(
    const Stmt &op,
    const std::vector<std::vector<std::pair<std::string, DepDirection>>> &cond,
    const FindDepsCallback &found, FindDepsMode mode, DepType depType,
    const FindDepsFilter &filter, bool ignoreReductionWAW) {
    if (cond.empty()) {
        return;
    }

    FindAccessPoint finder;
    finder(op);
    AnalyzeDeps analyzer(finder.points(), finder.reads(), finder.writes(),
                         finder.scope2coord(), cond, found, mode, depType,
                         filter, ignoreReductionWAW);
    analyzer(op);
}

} // namespace ir

