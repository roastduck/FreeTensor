#include <algorithm>
#include <climits>

#include <analyze/find_loop_variance.h>

namespace freetensor {

// to = from
template <class FromMap, class ToMap>
void copyInfo(const FromMap &fromInfo, const typename FromMap::key_type &from,
              ToMap &toInfo, const typename ToMap::key_type &to) {
    if (auto it = fromInfo.find(from); it != fromInfo.end()) {
        toInfo[to] = it->second;
    } else {
        toInfo.erase(to);
    }
}

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
            if (vari == LoopVariability::Variant || !toInfo.count(to) ||
                !toInfo.at(to).count(loop)) {
                // V + ? = V
                // ? + I = ?
                toInfo[to][loop] = vari;
            }
        }
    }
}

void InitExprVari::visitExpr(const Expr &expr) {
    BaseClass::visitExpr(expr);
    for (auto &&loop : loopStack_) {
        exprInfo_[{expr, curStmt()}][loop] = LoopVariability::Unknown;
    }
}

void InitExprVari::visit(const For &op) {
    loopStack_.emplace_back(op->id());
    BaseClass::visit(op);
    loopStack_.pop_back();
}

void MarkStores::meetTo(const Expr &from, const std::string &to) {
    ::freetensor::meetTo(exprInfo_, {from, curStmt()}, varInfo_, to);
}

void MarkStores::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    condStack_.emplace_back(op->len_); // May make a ReduceTo variant
    (*this)(op->body_);
    condStack_.pop_back();
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

int FindLoopVariance::unknownCnt() const {
    int ret = 0;
    for (auto &&[expr, info] : exprInfo_) {
        for (auto &&[loop, vari] : info) {
            if (vari == LoopVariability::Unknown) {
                ret++;
            }
        }
    }
    return ret;
}

void FindLoopVariance::copyInfo(const Expr &from, const Expr &to) {
    ::freetensor::copyInfo(exprInfo_, {from, curStmt()}, exprInfo_,
                           {to, curStmt()});
}

void FindLoopVariance::meetTo(const Expr &from, const Expr &to) {
    ::freetensor::meetTo(exprInfo_, {from, curStmt()}, exprInfo_,
                         {to, curStmt()});
}

void FindLoopVariance::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    loopStack_.emplace_back(op->id());
    condStack_.emplace_back(op->len_); // May make a ReduceTo variant
    loops_[op->iter_] = op;

    (*this)(op->body_);

    loops_.erase(op->iter_);
    condStack_.pop_back();
    loopStack_.pop_back();
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
    MarkStores{op->name_, condStack_, varInfo_, exprInfo_}(op);
    BaseClass::visit(op);
    ::freetensor::copyInfo(varInfo_, op->name_, uniqVarInfo_, op->id());
}

void FindLoopVariance::visitConst(const Const &op) {
    BaseClass::visitExpr(op);
    exprInfo_.erase({op, curStmt()}); // Invariant
}

void FindLoopVariance::visitBinOp(const BinaryExpr &op) {
    BaseClass::visitExpr(op);
    copyInfo(op->lhs_, op);
    meetTo(op->rhs_, op);
}

void FindLoopVariance::visitUnaryOp(const UnaryExpr &op) {
    BaseClass::visitExpr(op);
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
        BaseClass::visitExpr(op);
    }
}

void FindLoopVariance::visit(const Var &op) {
    BaseClass::visit(op);
    exprInfo_.erase({op, curStmt()});
    exprInfo_[{op, curStmt()}][loops_.at(op->name_)->id()] =
        LoopVariability::Variant;
}

void FindLoopVariance::visit(const Load &op) {
    BaseClass::visit(op);

    ::freetensor::copyInfo(varInfo_, op->var_, exprInfo_, {op, curStmt()});

    // varInfo_[op->var_] may contain info of non-surrounding loops, set them to
    // Invariant
    if (auto i = exprInfo_.find({op, curStmt()}); i != exprInfo_.end()) {
        for (auto j = i->second.begin(); j != i->second.end();) {
            if (std::find(loopStack_.begin(), loopStack_.end(), j->first) ==
                loopStack_.end()) {
                j = i->second.erase(j);
            } else {
                j++;
            }
        }
    }

    for (auto &&index : op->indices_) {
        meetTo(index, op);
    }
}

void FindLoopVariance::visit(const IfExpr &op) {
    BaseClass::visit(op);
    copyInfo(op->cond_, op);
    meetTo(op->thenCase_, op);
    meetTo(op->elseCase_, op);
}

void FindLoopVariance::visit(const Cast &op) {
    BaseClass::visit(op);
    copyInfo(op->expr_, op);
}

void FindLoopVariance::visit(const Intrinsic &op) {
    BaseClass::visit(op);
    if (op->hasSideEffect_) {
        for (auto &&loop : loopStack_) {
            exprInfo_[{op, curStmt()}][loop] = LoopVariability::Variant;
        }
    } else {
        for (auto &&[i, expr] : views::enumerate(op->params_)) {
            if (i == 0) {
                copyInfo(expr, op);
            } else {
                meetTo(expr, op);
            }
        }
    }
}

bool isVariant(const LoopVariExprMap &exprInfo, const StmtOrExprID &expr,
               const ID &loop) {
    return exprInfo.count(expr) && exprInfo.at(expr).count(loop);
}

bool isVariant(const LoopVariUniqVarMap &varInfo, const VarDef &def,
               const ID &loop) {
    return isVariant(varInfo, def->id(), loop);
}

bool isVariant(const LoopVariUniqVarMap &varInfo, const ID &defId,
               const ID &loop) {
    return varInfo.count(defId) && varInfo.at(defId).count(loop);
}

std::pair<LoopVariExprMap, LoopVariUniqVarMap>
findLoopVariance(const Stmt &op) {
    FindLoopVariance visitor(op);
    int lastCnt = INT_MAX;
    while (true) {
        visitor(op);
        int cnt = visitor.unknownCnt();
        ASSERT(cnt <= lastCnt);
        if (cnt == 0 || cnt == lastCnt) {
            return std::make_pair(visitor.exprInfo(), visitor.varInfo());
        }
        lastCnt = cnt;
    }
}

} // namespace freetensor
