#include <algorithm>
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
    loop2axis_[op->id()] = cur_.size();
    axis2loop_[cur_.size()] = op->id();
    cur_.emplace_back();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        cur_.back() = makeIntConst(i);
        (*this)(op->stmts_[i]);
    }
    cur_.pop_back();
}

void FindAccessPoint::visit(const For &op) {
    loop2axis_[op->id()] = cur_.size();
    axis2loop_[cur_.size()] = op->id();
    cur_.emplace_back(makeVar(op->iter_));
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op, defAxis_.at(op->var_), cur_, op->indices_};
    points_.emplace(op.get(), ap);
    reads_.emplace(op->var_, ap);
}

std::string AnalyzeDeps::linear2str(const LinearExpr &lin) const {
    std::ostringstream os;
    os << lin.bias_;
    for (auto &&item : lin.coeff_) {
        if (item.second.a->nodeType() == ASTNodeType::Var) {
            os << " + " << item.second.k << " " << item.second.a;
        } else {
            // TODO: Use the entire array as dependency
            ASSERT(false);
        }
    }
    return os.str();
}

std::string AnalyzeDeps::makeIterList(const std::vector<Expr> &list,
                                      int eraseBefore, int n) const {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i]->nodeType() == ASTNodeType::Var) {
                ret += list[i].as<VarNode>()->name_;
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
AnalyzeDeps::makeLinList(const std::vector<LinearExpr> &list) const {
    std::string ret;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        ret += linear2str(list[i]);
        if (i < iEnd - 1) {
            ret += ", ";
        }
    }
    return "[" + ret + "]";
}

std::string AnalyzeDeps::makeRange(const std::vector<Expr> &list) const {
    std::string ret;
    bool first = true;
    for (int i = 0, iEnd = list.size(); i < iEnd; i++) {
        if (list[i]->nodeType() == ASTNodeType::Var) {
            if (!first) {
                ret += " and ";
            }
            if (first) {
                first = false;
            }
            ret += "0 <= " + list[i].as<VarNode>()->name_ + " < N" +
                   std::to_string(i);
        }
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
    std::vector<LinearExpr> acc;
    acc.reserve(accDim);
    for (auto &&item : p.access_) {
        acc.emplace_back(linear_.at(item.get()));
    }
    auto ret = makeNdList("N", iterDim) + " -> {" +
               makeIterList(p.iter_, p.defAxis_, iterDim) + " -> " +
               makeLinList(acc) + ": " + makeRange(p.iter_) + "}";
    return ret;
}

std::string AnalyzeDeps::makeSingleIneq(FindDepsMode mode, int iterId,
                                        int iterDim) const {
    auto idStr = std::to_string(iterId);
    ASSERT(mode == FindDepsMode::Normal || mode == FindDepsMode::Inv);
    auto ineq = mode == FindDepsMode::Inv ? ">" : "<";
    return "{" + makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim) +
           ": d_" + idStr + " " + ineq + " d" + idStr + "}";
}

const std::string &AnalyzeDeps::getVar(const AST &op) {
    switch (op->nodeType()) {
    case ASTNodeType::Load:
        return op.as<LoadNode>()->var_;
    case ASTNodeType::Store:
        return op.as<StoreNode>()->var_;
    case ASTNodeType::AddTo:
        return op.as<AddToNode>()->var_;
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
            int iterId;
            FindDepsMode mode;
            std::tie(iterId, mode) = subitem;
            if (iterId >= iterDim) {
                found = false;
                break;
            }
            isl_basic_map *require = isl_basic_map_read_from_str(
                isl_, makeSingleIneq(mode, iterId, iterDim).c_str());
            isl_map *res = isl_map_intersect(isl_map_copy(nearest),
                                             isl_map_from_basic_map(require));
            found &= !isl_map_is_empty(res);
            isl_map_free(res);
        }
        if (found) {
            try {
                found_(item, getVar(point.op_), point.op_, other.op_);
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
    const Stmt &_op,
    const std::vector<std::vector<std::pair<std::string, FindDepsMode>>> &_cond,
    const std::function<
        void(const std::vector<std::pair<std::string, FindDepsMode>> &,
             const std::string &, const AST &, const AST &)> &_found) {
    auto op = Disambiguous()(_op);
    auto hash = getHashMap(op);
    AnalyzeLinear analyzeLinear(hash);
    analyzeLinear(op);
    auto &&linear = analyzeLinear.result();

    FindAccessPoint visitor;
    visitor(op);
    std::vector<std::vector<std::pair<int, FindDepsMode>>> cond;
    cond.reserve(_cond.size());
    for (auto &&item : _cond) {
        cond.emplace_back();
        cond.back().reserve(item.size());
        for (auto &&subitem : item) {
            cond.back().emplace_back(visitor.loop2axis().at(subitem.first),
                                     subitem.second);
        }
    }
    auto found = [&](const std::vector<std::pair<int, FindDepsMode>> &_cond,
                     const std::string &var, const AST &later,
                     const AST &earlier) {
        std::vector<std::pair<std::string, FindDepsMode>> cond;
        cond.reserve(_cond.size());
        for (auto &&item : _cond) {
            cond.emplace_back(visitor.axis2loop().at(item.first), item.second);
        }
        _found(cond, var, later, earlier);
    };
    AnalyzeDeps mutator(visitor.points(), visitor.reads(), visitor.writes(),
                        linear, cond, found);
    mutator(op);
}

} // namespace ir

