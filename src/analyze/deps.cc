#include <analyze/deps.h>
#include <analyze/hash.h>
#include <except.h>
#include <mutator.h>

namespace ir {

void FindAccessPoint::visit(const StmtSeq &op) {
    cur_.emplace_back();
    for (size_t i = 0, iEnd = op->stmts_.size(); i < iEnd; i++) {
        cur_.back() = makeIntConst(i);
        (*this)(op->stmts_[i]);
    }
    cur_.pop_back();
}

void FindAccessPoint::visit(const For &op) {
    if (!op->id_.empty()) {
        loop2axis_[op->id_] = cur_.size();
    }
    cur_.emplace_back(makeVar(op->iter_));
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const Store &op) {
    // For a[i] = a[i] + 1, write happens after read
    cur_.emplace_back(makeIntConst(0));
    auto ap = Ref<AccessPoint>::make();
    *ap = {op, cur_, op->indices_};
    points_.emplace(op.get(), ap);
    writes_.emplace(op->var_, ap);

    cur_.back() = makeIntConst(1);
    Visitor::visit(op);
    cur_.pop_back();
}

void FindAccessPoint::visit(const Load &op) {
    Visitor::visit(op);
    auto ap = Ref<AccessPoint>::make();
    *ap = {op, cur_, op->indices_};
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
                                      int n) const {
    std::string ret;
    for (int i = 0; i < n; i++) {
        if (i < (int)list.size()) {
            if (list[i]->nodeType() == ASTNodeType::Var) {
                ret += list[i].as<VarNode>()->name_;
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
    return makeNdList("N", iterDim) + " -> {" + makeIterList(p.iter_, iterDim) +
           " -> " + makeLinList(acc) + ": " + makeRange(p.iter_) + "}";
}

std::string AnalyzeDeps::makeSingleIneq(int iterId, int iterDim) const {
    auto idStr = std::to_string(iterId);
    return "{" + makeNdList("d", iterDim) + " -> " + makeNdList("d_", iterDim) +
           ": d_" + idStr + " > d" + idStr + "}";
}

void AnalyzeDeps::checkDep(const AccessPoint &point, const AccessPoint &other) {
    if (!permutable_) {
        return;
    }

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

    for (auto &&iterId : loops_) {
        if (iterId >= iterDim) {
            continue;
        }
        isl_basic_map *require = isl_basic_map_read_from_str(
            isl_, makeSingleIneq(iterId, iterDim).c_str());
        isl_map *res = isl_map_intersect(isl_map_copy(nearest),
                                         isl_map_from_basic_map(require));

        permutable_ &= isl_map_is_empty(res);
        isl_map_free(res);
        if (!permutable_) {
            msg_ << "Dependency "
                 << (point.op_->nodeType() == ASTNodeType::Load ? "READ "
                                                                : "WRITE ")
                 << point.op_ << " after "
                 << (other.op_->nodeType() == ASTNodeType::Load ? "READ "
                                                                : "WRITE ")
                 << other.op_ << " cannot be resolved";
            break;
        }
    }
    isl_map_free(nearest);
}

void AnalyzeDeps::visit(const Store &op) {
    Visitor::visit(op);
    auto &&point = points_.at(op.get());
    auto range = reads_.equal_range(op->var_);
    for (auto i = range.first; i != range.second; i++) {
        checkDep(*point, *(i->second)); // WAR
    }
    range = writes_.equal_range(op->var_);
    for (auto i = range.first; i != range.second; i++) {
        checkDep(*point, *(i->second)); // WAW
    }
}

void AnalyzeDeps::visit(const Load &op) {
    Visitor::visit(op);
    auto &&point = points_.at(op.get());
    auto range = writes_.equal_range(op->var_);
    for (auto i = range.first; i != range.second; i++) {
        checkDep(*point, *(i->second)); // RAW
    }
}

std::pair<bool, std::string>
isPermutable(const Stmt &_op, const std::vector<std::string> loops) {
    auto op = Disambiguous()(_op);
    auto hash = getHashMap(op);
    AnalyzeLinear analyzeLinear(hash);
    analyzeLinear(op);
    auto &&linear = analyzeLinear.result();

    FindAccessPoint visitor;
    visitor(op);
    std::vector<int> loopIds;
    loopIds.reserve(loops.size());
    for (auto &&item : loops) {
        loopIds.emplace_back(visitor.loop2axis().at(item));
    }
    AnalyzeDeps mutator(visitor.points(), visitor.reads(), visitor.writes(),
                        loopIds, linear);
    mutator(op);
    return std::make_pair(mutator.permutable(), mutator.msg());
}

} // namespace ir

