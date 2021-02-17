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
    cur_.emplace_back();
    begin_.emplace_back(makeIntConst(0));
    end_.emplace_back(makeIntConst(op->stmts_.size()));
    scope2coord_[op->id()] = cur_;
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        cur_.back() = makeIntConst(i);
        (*this)(op->stmts_[i]);
    }
    cur_.pop_back();
    begin_.pop_back();
    end_.pop_back();
}

void FindAccessPoint::visit(const For &op) {
    cur_.emplace_back(makeVar(op->iter_));
    begin_.emplace_back(op->begin_);
    end_.emplace_back(op->end_);
    scope2coord_[op->id()] = cur_;
    Visitor::visit(op);
    cur_.pop_back();
    begin_.pop_back();
    end_.pop_back();
}

void FindAccessPoint::visit(const If &op) {
    (*this)(op->cond_);
    auto oldCond = cond_;
    cond_ = oldCond.isValid() ? makeLAnd(oldCond, op->cond_) : op->cond_;
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        cond_ = oldCond.isValid() ? makeLAnd(oldCond, makeLNot(op->cond_))
                                  : makeLNot(op->cond_);
        (*this)(op->elseCase_);
    }
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    ASSERT(cur_.size() == begin_.size());
    ASSERT(cur_.size() == end_.size());
    *ap = {op,     cursor(), defAxis_.at(op->var_), cur_,
           begin_, end_,     op->indices_,          cond_};
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

std::string AnalyzeDeps::makeIterList(const std::vector<Expr> &list,
                                      int eraseBefore, int n) {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i]->nodeType() == ASTNodeType::Var) {
                ret += genISLExpr_.normalizeId(list[i].as<VarNode>()->name_);
                if (i < eraseBefore) {
                    ret += " = 0";
                }
            } else if (list[i]->nodeType() == ASTNodeType::IntConst) {
                ret += std::to_string(list[i].as<IntConstNode>()->val_);
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

std::string AnalyzeDeps::makeAccList(const std::vector<Expr> &list) {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (auto linstr = genISLExpr_(list[i]); linstr.isValid()) {
            ret += *linstr;
        } else {
            ret += genISLExpr_.normalizeId("free" + std::to_string(i));
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

std::string AnalyzeDeps::makeRange(const std::vector<Expr> &point,
                                   const std::vector<Expr> &begin,
                                   const std::vector<Expr> &end) {
    size_t n = point.size();
    ASSERT(begin.size() == n);
    ASSERT(end.size() == n);
    std::vector<std::string> ineqs;
    for (size_t i = 0; i < n; i++) {
        if (point[i]->nodeType() == ASTNodeType::Var) {
            std::string ineq =
                genISLExpr_.normalizeId(point[i].as<VarNode>()->name_);
            bool bounded = false;
            if (auto linstr = genISLExpr_(begin[i]); linstr.isValid()) {
                ineq = *linstr + " <= " + ineq;
                bounded = true;
            }
            if (auto linstr = genISLExpr_(end[i]); linstr.isValid()) {
                ineq = ineq + " < " + *linstr;
                bounded = true;
            }
            if (bounded) {
                ineqs.emplace_back(std::move(ineq));
            }
        }
    }
    std::string ret;
    for (size_t i = 0, iEnd = ineqs.size(); i < iEnd; i++) {
        if (i > 0) {
            ret += " and ";
        }
        ret += ineqs[i];
    }
    return ret;
}

std::string AnalyzeDeps::makeCond(const Expr &expr) {
    if (expr.isValid()) {
        // TODO: Try to eliminate LNot. But how to simplify a single expression?
        // auto expr = simplifyPass(_expr);
        if (auto str = genISLExpr_(expr); str.isValid()) {
            return " and " + *str;
        }
    }
    return "";
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

std::string AnalyzeDeps::makeAccMap(const AccessPoint &p, int iterDim,
                                    int accDim) {
    return "{" + makeIterList(p.iter_, p.defAxis_, iterDim) + " -> " +
           makeAccList(p.access_) + ": " +
           makeRange(p.iter_, p.begin_, p.end_) + makeCond(p.cond_) + "}";
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

std::string AnalyzeDeps::makeIneqBetweenOps(FindDepsMode mode, int iterId,
                                            int iterDim) const {
    auto idStr = std::to_string(iterId);
    std::string ineq;
    switch (mode) {
    case FindDepsMode::Inv:
        ineq = ">";
        break;
    case FindDepsMode::Same:
        ineq = "=";
        break;
    case FindDepsMode::Normal:
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

    isl_basic_map *pmap = isl_basic_map_read_from_str(
        isl_, makeAccMap(point, iterDim, accDim).c_str());
    isl_basic_map *omap = isl_basic_map_read_from_str(
        isl_, makeAccMap(other, iterDim, accDim).c_str());
    isl_basic_map *depall =
        isl_basic_map_apply_range(pmap, isl_basic_map_reverse(omap));

    isl_basic_set *domain = isl_basic_set_read_from_str(
        isl_, ("{" + makeNdList("d", iterDim) + "}").c_str());
    isl_space *space = isl_basic_set_get_space(domain);
    isl_basic_set_free(domain);
    isl_map *pred = isl_map_lex_gt(space);

    isl_map *dep = isl_map_intersect(isl_map_from_basic_map(depall), pred);
    isl_map *nearest = isl_map_lexmax(dep);

    for (auto &&item : cond_) {
        bool found = true;
        for (auto &&subitem : item) {
            auto &&coord = scope2coord_.at(subitem.first);
            int iterId = coord.size() - 1;
            FindDepsMode mode = subitem.second;
            if (iterId >= iterDim) {
                found = false;
                break;
            }
            isl_map *res = isl_map_copy(nearest);

            // Position in the outer StmtSeq nodes
            std::vector<std::pair<int, int>> pos;
            for (int i = 0; i < iterId; i++) {
                if (coord[i]->nodeType() == ASTNodeType::IntConst) {
                    pos.emplace_back(i, coord[i].as<IntConstNode>()->val_);
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
                found_(item, getVar(point.op_), point.op_, other.op_,
                       point.cursor_, other.cursor_);
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
    auto &&point = points_.at(op);
    auto range = writes_.equal_range(op->var_);
    for (auto i = range.first; i != range.second; i++) {
        checkDep(*point, *(i->second)); // RAW
    }
}

void findDeps(
    const Stmt &op,
    const std::vector<std::vector<std::pair<std::string, FindDepsMode>>> &cond,
    const FindDepsCallback &found, bool ignoreReductionWAW) {
    ASSERT(op->noAmbiguous());

    FindAccessPoint visitor;
    visitor(op);
    AnalyzeDeps mutator(visitor.points(), visitor.reads(), visitor.writes(),
                        visitor.scope2coord(), cond, found, ignoreReductionWAW);
    mutator(op);
}

} // namespace ir

