#include <algorithm>
#include <sstream>

#include <analyze/deps.h>
#include <analyze/normalize.h>
#include <except.h>
#include <mutator.h>
#include <pass/disambiguous.h>
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
    ASSERT(op->infoNotCond_.isValid());
    (*this)(op->infoNotCond_);

    auto oldCond = cond_;
    cond_ = oldCond.isValid() ? makeLAnd(oldCond, op->cond_) : op->cond_;
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        cond_ = oldCond.isValid() ? makeLAnd(oldCond, op->infoNotCond_)
                                  : op->infoNotCond_;
        (*this)(op->elseCase_);
    }
    cond_ = oldCond;
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op, cursor(), defAxis_.at(op->var_), cur_, op->indices_, cond_};
    points_.emplace(op, ap);
    reads_.emplace(op->var_, ap);
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

Ref<std::string> GenISLExpr::linear2str(const LinearExpr &lin) {
    std::ostringstream os;
    os << lin.bias_;
    for (auto &&item : lin.coeff_) {
        if (item.second.a->nodeType() == ASTNodeType::Var) {
            os << " + " << item.second.k << " "
               << normalizeId(toString(item.second.a));
        } else {
            // Use the entire array as dependency
            return nullptr;
        }
    }
    return Ref<std::string>::make(os.str());
}

Ref<std::string> GenISLExpr::operator()(const Expr &op) {
    std::vector<LinearExpr> subexprs;
    std::function<bool(const Expr &expr)> recur = [&](const Expr &expr) {
        if (expr->nodeType() == ASTNodeType::LAnd) {
            auto a = expr.as<LAndNode>();
            return recur(a->lhs_) && recur(a->rhs_);
        }
        analyzeLinear_(expr);
        if (analyzeLinear_.result().count(expr)) {
            subexprs.emplace_back(analyzeLinear_.result().at(expr));
            return true;
        }
        return false;
    };
    if (!recur(op)) {
        return nullptr;
    }
    std::string ret;
    for (auto &&sub : subexprs) {
        if (!ret.empty()) {
            ret += " and ";
        }
        if (auto str = linear2str(sub); str.isValid()) {
            ret += *str;
        } else {
            return nullptr;
        }
    }
    return Ref<std::string>::make(std::move(ret));
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
        if (auto linstr = genISLExpr_(list[i]); linstr.isValid()) {
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
            int bounded = 0;
            if (auto linstr = genISLExpr_(point[i].begin_); linstr.isValid()) {
                ineq = *linstr + " <= " + ineq;
                bounded++;
            }
            if (auto linstr = genISLExpr_(point[i].end_); linstr.isValid()) {
                ineq = ineq + " < " + *linstr;
                bounded++;
            }
            if (bounded < 2 && relax == RelaxMode::Necessary) {
                return nullptr;
            }
            if (bounded > 0) {
                ineqs.emplace_back(std::move(ineq));
            }
        }
    }
    std::string ret;
    for (size_t i = 0, iEnd = ineqs.size(); i < iEnd; i++) {
        ret += i == 0 ? ": " : " and ";
        ret += ineqs[i];
    }
    return Ref<std::string>::make(std::move(ret));
}

Ref<std::string> AnalyzeDeps::makeCond(const Expr &expr, RelaxMode relax) {
    if (expr.isValid()) {
        if (auto str = genISLExpr_(expr); str.isValid()) {
            return Ref<std::string>::make(" and " + *str);
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
    if (auto str = makeRange(p.iter_, relax); str.isValid()) {
        ret += *str;
    } else {
        return nullptr;
    }
    if (auto str = makeCond(p.cond_, relax); str.isValid()) {
        ret += *str;
    } else {
        return nullptr;
    }
    return Ref<std::string>::make("{" + ret + "}");
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

    isl_basic_map *pmap, *omap;
    if (auto str = makeAccMap(point, iterDim, accDim, pRelax); str.isValid()) {
        pmap = isl_basic_map_read_from_str(isl_, str->c_str());
    } else {
        return;
    }
    if (auto str = makeAccMap(other, iterDim, accDim, oRelax); str.isValid()) {
        omap = isl_basic_map_read_from_str(isl_, str->c_str());
    } else {
        isl_basic_map_free(pmap);
        return;
    }

    isl_basic_set *domain = isl_basic_set_read_from_str(
        isl_, ("{" + makeNdList("d", iterDim) + "}").c_str());
    isl_space *space = isl_basic_set_get_space(domain);
    isl_basic_set_free(domain);
    isl_map *pred = isl_map_lex_gt(space);

    isl_set *pIter =
        isl_set_from_basic_set(isl_basic_map_domain(isl_basic_map_copy(omap)));
    isl_basic_map *depall =
        isl_basic_map_apply_range(pmap, isl_basic_map_reverse(omap));
    isl_map *dep = isl_map_intersect(isl_map_from_basic_map(depall), pred);
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
            checkDep(*point, *(i->second));
        }
    }
}

Stmt prepareFindDeps(const Stmt &_op) {
    auto op = normalize(_op); // for IfNode::infoNotCond_
    op = simplifyPass(op);
    op = disambiguous(op);
    return op;
}

void findDeps(
    const Stmt &op,
    const std::vector<std::vector<std::pair<std::string, DepDirection>>> &cond,
    const FindDepsCallback &found, FindDepsMode mode, DepType depType,
    bool ignoreReductionWAW) {
    ASSERT(op->noAmbiguous());

    FindAccessPoint finder;
    finder(op);
    AnalyzeDeps analyzer(finder.points(), finder.reads(), finder.writes(),
                         finder.scope2coord(), cond, found, mode, depType,
                         ignoreReductionWAW);
    analyzer(op);
}

} // namespace ir

