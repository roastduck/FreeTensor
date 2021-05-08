#include <analyze/find_all_loops.h>
#include <analyze/find_loop_variance.h>

namespace ir {

void MarkStores::mergeInfo(const Expr &from, const std::string &to) {
    if (exprInfo_.count(from)) {
        for (auto &&loop : exprInfo_.at(from)) {
            if (loop.second == LoopVariability::Variance) {
                varInfo_[to][loop.first] = LoopVariability::Variance;
            }
        }
    }
    if (varInfo_.count(to)) {
        std::vector<std::string> toErase;
        for (auto &&loop : varInfo_.at(to)) {
            if (!exprInfo_.count(from) ||
                !exprInfo_.at(from).count(loop.first)) {
                toErase.emplace_back(loop.first);
            }
        }
        for (auto &&item : toErase) {
            varInfo_.at(to).erase(item);
        }
    }
}

void MarkStores::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    varInfo_[var_][op->id()] = LoopVariability::Invariance;

    varInfo_[op->iter_][op->id()] = LoopVariability::Variance;
    for (auto &&loop : loopStack_) {
        varInfo_[op->iter_][loop->id()] = LoopVariability::Invariance;
        varInfo_[loop->iter_][op->id()] = LoopVariability::Invariance;
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

void FindLoopVariance::copyInfo(const Expr &from, const Expr &to) {
    if (exprInfo_.count(from)) {
        exprInfo_[to] = exprInfo_.at(from);
    }
}

void FindLoopVariance::mergeInfo(const Expr &from, const Expr &to) {
    if (exprInfo_.count(from)) {
        for (auto &&loop : exprInfo_.at(from)) {
            if (loop.second == LoopVariability::Variance) {
                exprInfo_[to][loop.first] = LoopVariability::Variance;
            }
        }
    }
    if (exprInfo_.count(to)) {
        std::vector<std::string> toErase;
        for (auto &&loop : exprInfo_.at(to)) {
            if (!exprInfo_.count(from) ||
                !exprInfo_.at(from).count(loop.first)) {
                toErase.emplace_back(loop.first);
            }
        }
        for (auto &&item : toErase) {
            exprInfo_.at(to).erase(item);
        }
    }
}

void FindLoopVariance::visit(const For &op) {
    (*this)(op->begin_);
    (*this)(op->end_);
    (*this)(op->len_);

    varInfo_[op->iter_][op->id()] = LoopVariability::Variance;
    for (auto &&loop : loopStack_) {
        varInfo_[op->iter_][loop->id()] = LoopVariability::Invariance;
        varInfo_[loop->iter_][op->id()] = LoopVariability::Invariance;
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

    varInfo_.erase(op->name_);
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
        mergeInfo(index, op);
    }
}

void FindLoopVariance::visit(const LNot &op) {
    Visitor::visit(op);
    copyInfo(op->expr_, op);
}

bool isVariant(
    const std::unordered_map<
        Expr, std::unordered_map<std::string, LoopVariability>> &exprInfo,
    const Expr &expr, const std::string &loop) {
    if (!exprInfo.count(expr)) {
        return true;
    }
    if (!exprInfo.at(expr).count(loop)) {
        return true;
    }
    return exprInfo.at(expr).at(loop) == LoopVariability::Variance;
}

std::unordered_map<Expr, std::unordered_map<std::string, LoopVariability>>
findLoopVariance(const AST &op) {
    FindAllLoops finder;
    finder(op);
    FindLoopVariance visitor(finder.loops());
    int lastCnt = 0;
    while (true) {
        visitor(op);
        int cnt = visitor.exprInfo().size();
        if (cnt == lastCnt) {
            return visitor.exprInfo();
        }
        lastCnt = cnt;
    }
}

} // namespace ir

