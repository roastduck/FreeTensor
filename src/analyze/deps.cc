#include <algorithm>
#include <regex>
#include <sstream>

#include <analyze/deps.h>
#include <analyze/hash.h>
#include <except.h>
#include <mutator.h>

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

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    ASSERT(cur_.size() == begin_.size());
    ASSERT(cur_.size() == end_.size());
    *ap = {op,     cursor(), defAxis_.at(op->var_), cur_,
           begin_, end_,     op->indices_};
    points_.emplace(op.get(), ap);
    reads_.emplace(op->var_, ap);
}

std::string AnalyzeDeps::normalizeId(const std::string &id) const {
    return std::regex_replace(id, std::regex("\\."), "__dot__");
}

Ref<std::string> AnalyzeDeps::linear2str(const LinearExpr &lin) const {
    std::ostringstream os;
    os << lin.bias_;
    for (auto &&item : lin.coeff_) {
        if (item.second.a->nodeType() == ASTNodeType::Var) {
            os << " + " << item.second.k << " " << item.second.a;
        } else {
            // Use the entire array as dependency
            return nullptr;
        }
    }
    return Ref<std::string>::make(normalizeId(os.str()));
}

std::string AnalyzeDeps::makeIterList(const std::vector<Expr> &list,
                                      int eraseBefore, int n) const {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i]->nodeType() == ASTNodeType::Var) {
                ret += normalizeId(list[i].as<VarNode>()->name_);
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

std::string
AnalyzeDeps::makeLinList(const std::vector<Ref<LinearExpr>> &list) const {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (list[i].isValid()) {
            if (auto linstr = linear2str(*list[i]); linstr.isValid()) {
                ret += *linstr;
            } else {
                ret += "free" + std::to_string(i);
            }
        } else {
            ret += "free" + std::to_string(i);
        }
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

std::string AnalyzeDeps::makeRange(const std::vector<Expr> &point,
                                   const std::vector<Expr> &begin,
                                   const std::vector<Expr> &end) const {
    size_t n = point.size();
    ASSERT(begin.size() == n);
    ASSERT(end.size() == n);
    std::vector<std::string> ineqs;
    for (size_t i = 0; i < n; i++) {
        if (point[i]->nodeType() == ASTNodeType::Var) {
            std::string ineq = normalizeId(point[i].as<VarNode>()->name_);
            bool bounded = false;
            if (linear_.count(begin[i].get())) {
                if (auto linstr = linear2str(linear_.at(begin[i].get()));
                    linstr.isValid()) {
                    ineq = *linstr + " <= " + ineq;
                    bounded = true;
                }
            }
            if (linear_.count(end[i].get())) {
                if (auto linstr = linear2str(linear_.at(end[i].get()));
                    linstr.isValid()) {
                    ineq = ineq + " < " + *linstr;
                    bounded = true;
                }
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
                                    int accDim) const {
    std::vector<Ref<LinearExpr>> acc;
    acc.reserve(accDim);
    for (auto &&item : p.access_) {
        if (linear_.count(item.get())) {
            acc.emplace_back(Ref<LinearExpr>::make(linear_.at(item.get())));
        } else {
            acc.emplace_back(nullptr);
        }
    }
    auto ret = "{" + makeIterList(p.iter_, p.defAxis_, iterDim) + " -> " +
               makeLinList(acc) + ": " + makeRange(p.iter_, p.begin_, p.end_) +
               "}";
    return ret;
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
    auto &&point = points_.at(op.get());
    auto range = writes_.equal_range(op->var_);
    for (auto i = range.first; i != range.second; i++) {
        checkDep(*point, *(i->second)); // RAW
    }
}

void findDeps(
    const Stmt &op,
    const std::vector<std::vector<std::pair<std::string, FindDepsMode>>> &cond,
    const FindDepsCallback &found) {
    ASSERT(op->noAmbiguous());
    auto hash = getHashMap(op);
    AnalyzeLinear analyzeLinear(hash);
    analyzeLinear(op);
    auto &&linear = analyzeLinear.result();

    FindAccessPoint visitor;
    visitor(op);
    AnalyzeDeps mutator(visitor.points(), visitor.reads(), visitor.writes(),
                        visitor.scope2coord(), linear, cond, found);
    mutator(op);
}

} // namespace ir

