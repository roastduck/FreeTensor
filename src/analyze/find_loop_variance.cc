#include <analyze/find_all_loops.h>
#include <analyze/find_loop_variance.h>

namespace freetensor {

// to = from meet to
template <class FromMap, class ToMap>
void meetTo(const FromMap &fromInfo, const typename FromMap::key_type &from,
            ToMap &toInfo, const typename ToMap::key_type &to) {
    //     I U V
    //   +-------
    // I | I U V
    // U | U U V
    // V | V V V
    if (fromInfo.count(from)) {
        for (auto &&[loop, vari] : fromInfo.at(from)) {
            if (vari == LoopVariability::Variant) {
                // V + ? = V
                toInfo[to][loop] = LoopVariability::Variant;
            }
        }
    }
    if (auto i = toInfo.find(to); i != toInfo.end()) {
        for (auto j = i->second.begin(); j != i->second.end();) {
            auto &&[loop, vari] = *j;
            if (vari == LoopVariability::Invariant) {
                // ? + I = ?
                if (!fromInfo.count(from) || !fromInfo.at(from).count(loop)) {
                    j = i->second.erase(j);
                    continue;
                } else {
                    j->second = fromInfo.at(from).at(loop);
                }
            }
            j++;
        }
        if (i->second.empty()) {
            i = toInfo.erase(i);
        }
    }
}

void MarkStores::meetTo(const Expr &from, const std::string &to) {
    ::freetensor::meetTo(exprInfo_, from, varInfo_, to);
}

void MarkStores::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    varInfo_[var_][op->id()] = LoopVariability::Invariant;

    varInfo_[op->iter_][op->id()] = LoopVariability::Variant;
    for (auto &&loop : loopStack_) {
        varInfo_[op->iter_][loop->id()] = LoopVariability::Invariant;
        varInfo_[loop->iter_][op->id()] = LoopVariability::Invariant;
    }
    loopStack_.emplace_back(op);
    condStack_.emplace_back(op->len_); // May make a ReduceTo variant

    (*this)(op->body_);

    condStack_.pop_back();
    loopStack_.pop_back();
    varInfo_.erase(op->iter_);
}

void MarkStores::visit(const If &op) {
    (*this)(op->cond_);

    condStack_.emplace_back(op->cond_);
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        (*this)(op->elseCase_);
    }
    condStack_.pop_back();
}

int FindLoopVariance::knownCnt() const {
    int ret = 0;
    for (auto &&loop : exprInfo_) {
        ret += loop.second.size();
    }
    return ret;
}

void FindLoopVariance::copyInfo(const Expr &from, const Expr &to) {
    if (exprInfo_.count(from)) {
        exprInfo_[to] = exprInfo_.at(from);
    }
}

void FindLoopVariance::meetTo(const Expr &from, const Expr &to) {
    ::freetensor::meetTo(exprInfo_, from, exprInfo_, to);
}

void FindLoopVariance::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    varInfo_[op->iter_][op->id()] = LoopVariability::Variant;
    for (auto &&loop : loopStack_) {
        varInfo_[op->iter_][loop->id()] = LoopVariability::Invariant;
        varInfo_[loop->iter_][op->id()] = LoopVariability::Invariant;
    }
    loopStack_.emplace_back(op);
    condStack_.emplace_back(op->len_); // May make a ReduceTo variant

    (*this)(op->body_);

    condStack_.pop_back();
    loopStack_.pop_back();
    varInfo_.erase(op->iter_);
}

void FindLoopVariance::visit(const If &op) {
    (*this)(op->cond_);

    condStack_.emplace_back(op->cond_);
    (*this)(op->thenCase_);
    if (op->elseCase_.isValid()) {
        (*this)(op->elseCase_);
    }
    condStack_.pop_back();
}

void FindLoopVariance::visit(const VarDef &op) {
    MarkStores{op->name_, loopStack_, condStack_, varInfo_, exprInfo_}(op);

    Visitor::visit(op);

    if (varInfo_.count(op->name_)) {
        uniqVarInfo_[op] = varInfo_.at(op->name_);
    }
    varInfo_.erase(op->name_);
}

void FindLoopVariance::visitConst(const Const &op) {
    Visitor::visitExpr(op);
    exprInfo_[op].reserve(allLoops_.size());
    for (auto &&loop : allLoops_) {
        exprInfo_[op][loop] = LoopVariability::Invariant;
    }
}

void FindLoopVariance::visitBinOp(const BinaryExpr &op) {
    Visitor::visitExpr(op);
    copyInfo(op->lhs_, op);
    meetTo(op->rhs_, op);
}

void FindLoopVariance::visitUnaryOp(const UnaryExpr &op) {
    Visitor::visitExpr(op);
    copyInfo(op->expr_, op);
}

void FindLoopVariance::visitExpr(const Expr &op) {
    if (op->isConst()) {
        visitConst(op.as<ConstNode>());
    } else if (op->isBinary()) {
        visitBinOp(op.as<BinaryExprNode>());
    } else if (op->isUnary()) {
        visitUnaryOp(op.as<UnaryExprNode>());
    } else {
        Visitor::visitExpr(op);
    }
}

void FindLoopVariance::visit(const Var &op) {
    Visitor::visit(op);
    if (varInfo_.count(op->name_)) {
        exprInfo_[op] = varInfo_.at(op->name_);
    }
}

void FindLoopVariance::visit(const Load &op) {
    Visitor::visit(op);
    if (varInfo_.count(op->var_)) {
        exprInfo_[op] = varInfo_.at(op->var_);
    }
    for (auto &&index : op->indices_) {
        meetTo(index, op);
    }
}

void FindLoopVariance::visit(const IfExpr &op) {
    Visitor::visit(op);
    copyInfo(op->cond_, op);
    meetTo(op->thenCase_, op);
    meetTo(op->elseCase_, op);
}

void FindLoopVariance::visit(const Cast &op) {
    Visitor::visit(op);
    copyInfo(op->expr_, op);
}

bool isVariant(const LoopVariExprMap &exprInfo, const Expr &expr,
               const ID &loop) {
    if (!exprInfo.count(expr)) {
        return true;
    }
    if (!exprInfo.at(expr).count(loop)) {
        return true;
    }
    return exprInfo.at(expr).at(loop) == LoopVariability::Variant;
}

bool isVariant(const LoopVariUniqVarMap &varInfo, const VarDef &def,
               const ID &loop) {
    if (!varInfo.count(def)) {
        return true;
    }
    if (!varInfo.at(def).count(loop)) {
        return true;
    }
    return varInfo.at(def).at(loop) == LoopVariability::Variant;
}

std::pair<LoopVariExprMap, LoopVariUniqVarMap>
findLoopVariance(const Stmt &op) {
    auto allLoops = findAllLoops(op);
    FindLoopVariance visitor(allLoops);
    int lastCnt = 0;
    while (true) {
        visitor(op);
        int cnt = visitor.knownCnt();
        if (cnt == lastCnt) {
            return std::make_pair(visitor.exprInfo(), visitor.varInfo());
        }
        lastCnt = cnt;
    }
}

} // namespace freetensor
