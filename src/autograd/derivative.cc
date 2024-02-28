#include <analyze/all_uses.h>
#include <autograd/derivative.h>
#include <autograd/replace_by_saved.h>
#include <hash.h>

namespace freetensor {

bool Derivative::LazyPartialDerivative::usingStore() {
    if (!usingStore_.has_value()) {
        if (invertFromStore_.has_value()) {
            usingStore_ = invertFromStore_->find(mathExpr_);
        } else {
            usingStore_ = false;
        }
    }
    return *usingStore_;
}

const std::unordered_set<std::string> &
Derivative::LazyPartialDerivative::reads() {
    if (!reads_.has_value()) {
        if (invertFromStore_.has_value()) {
            reads_ = invertFromStore_->allReadsExcludingInversion(mathExpr_);
        } else {
            reads_ = allReads(mathExpr_);
        }
    }
    return *reads_;
}

Expr Derivative::LazyPartialDerivative::replaceExpr(
    const std::unordered_map<ID, std::string> &intermediatesMap,
    const std::unordered_map<StmtOrExprID, Expr> &versions, const Expr &expr) {
    return ReplaceBySaved{symbolTable_, intermediatesMap, versions, rootStmtID_,
                          invertFromStore_}
        .grad(expr);
}

void Derivative::LazyFullDerivative::addPartial(
    const Load &x, const LazyPartialDerivative &partial) {
    usingStore_ = std::nullopt;
    for (auto &[l, p] : partials_) {
        if (HashComparator{}(l, x)) {
            p.merge(partial);
            return;
        }
    }
    partials_.emplace_back(x, partial);
}

bool Derivative::LazyFullDerivative::usingStore() {
    if (!usingStore_.has_value()) {
        for (auto &&[_, partial] : partials_) {
            if (partial.usingStore()) {
                usingStore_ = true;
                goto done;
            }
        }
        usingStore_ = false;
    done:;
    }
    return *usingStore_;
}

const std::unordered_set<std::string> &Derivative::LazyFullDerivative::reads() {
    if (!reads_.has_value()) {
        std::unordered_set<std::string> reads;
        for (auto &&[_, partial] : partials_) {
            for (auto &&name : partial.reads()) {
                reads.emplace(name);
            }
        }
        reads_ = std::move(reads);
    }
    return *reads_;
}

std::vector<Stmt> Derivative::LazyFullDerivative::genGrads(
    const std::unordered_map<ID, std::string> &intermediatesMap,
    const std::unordered_map<StmtOrExprID, Expr> &versions,
    const std::unordered_map<std::string, std::string> &gradNames,
    const Expr &gradY) {
    if (error_) {
        std::rethrow_exception(error_);
    }
    std::vector<Stmt> stmts;
    for (auto &&[load, _derivativeLazy] : partials_) {
        auto &&derivativeLazy = _derivativeLazy;
        auto &&derivative =
            derivativeLazy.genReplaced(intermediatesMap, versions);
        if (auto it = gradNames.find(load->var_); it != gradNames.end()) {
            auto indices = ranges::to<std::vector>(
                load->indices_ | views::transform([&](const Expr &idx) {
                    return derivativeLazy.replaceExpr(intermediatesMap,
                                                      versions, idx);
                }));
            auto expr = makeMul(gradY, derivative);
            if (upCast(load->loadType_, expr->dtype()).base() !=
                load->loadType_.base()) {
                expr = makeCast(expr, load->loadType_.base());
            }
            stmts.emplace_back(makeReduceTo(it->second, std::move(indices),
                                            ReduceOp::Add, std::move(expr),
                                            false));
        }
    }
    return stmts;
}

void Derivative::setPartial(const Expr &expr, const Expr &partial) {
    if (isFloat(expr->dtype())) {
        partials_[expr] =
            LazyPartialDerivative{symbolTableSnapshot(), partial,
                                  rootExpr_.stmtId(), invertFromStore_};
    }
}

void Derivative::visitExpr(const Expr &expr) {
    if (!isFloat(expr->dtype())) {
        return;
    }
    if (!rootExpr_.isValid()) {
        rootExpr_ = StmtOrExprID{expr, expr->parentStmt()};
        setPartial(expr, makeIntConst(1));
        try {
            BaseClass::visitExpr(expr);
        } catch (const InvalidAutoGrad &e) {
            derivatives_[rootExpr_].setError(std::current_exception());
        }
        rootExpr_ = StmtOrExprID{};
    } else {
        BaseClass::visitExpr(expr);
    }
}

void Derivative::visit(const Store &op) {
    invertFromStore_ = {op, op->expr_, [](const Expr &e) { return e; }};
    BaseClass::visit(op);
    invertFromStore_ = std::nullopt;
}

void Derivative::visit(const Load &op) {
    // No need to recurse into indices
    if (auto it = partials_.find(op); it != partials_.end()) {
        derivatives_[rootExpr_].addPartial(op, it->second);
    }
}

void Derivative::visit(const Add &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, it->second.mathExpr());
        setPartial(op->rhs_, it->second.mathExpr());
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->rhs_).empty()) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->lhs_, [=](const Expr &e) {
                return makeSub(oldInvertFromStore->invert(e), op->rhs_);
            }};
        (*this)(op->lhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->lhs_);
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->lhs_).empty()) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->rhs_, [=](const Expr &e) {
                return makeSub(oldInvertFromStore->invert(e), op->lhs_);
            }};
        (*this)(op->rhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->rhs_);
    }
}

void Derivative::visit(const Sub &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, it->second.mathExpr());
        setPartial(op->rhs_, makeSub(makeIntConst(0), it->second.mathExpr()));
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->rhs_).empty()) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->lhs_, [=](const Expr &e) {
                return makeAdd(oldInvertFromStore->invert(e), op->rhs_);
            }};
        (*this)(op->lhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->lhs_);
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->lhs_).empty()) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->rhs_, [=](const Expr &e) {
                return makeSub(op->lhs_, oldInvertFromStore->invert(e));
            }};
        (*this)(op->rhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->rhs_);
    }
}

void Derivative::visit(const Mul &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, makeMul(it->second.mathExpr(), op->rhs_));
        setPartial(op->rhs_, makeMul(it->second.mathExpr(), op->lhs_));
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->rhs_).empty() && isNE0(op->rhs_->dtype())) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->lhs_, [=](const Expr &e) {
                return makeRealDiv(oldInvertFromStore->invert(e), op->rhs_);
            }};
        (*this)(op->lhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->lhs_);
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->lhs_).empty() && isNE0(op->lhs_->dtype())) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->rhs_, [=](const Expr &e) {
                return makeRealDiv(oldInvertFromStore->invert(e), op->lhs_);
            }};
        (*this)(op->rhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->rhs_);
    }
}

void Derivative::visit(const RealDiv &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, makeRealDiv(it->second.mathExpr(), op->rhs_));
        setPartial(op->rhs_, makeSub(makeIntConst(0),
                                     makeMul(it->second.mathExpr(),
                                             makeRealDiv(op, op->rhs_))));
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->rhs_).empty()) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->lhs_, [=](const Expr &e) {
                return makeMul(oldInvertFromStore->invert(e), op->rhs_);
            }};
        (*this)(op->lhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->lhs_);
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op) &&
        allReads(op->lhs_).empty() && isNE0(op->dtype())) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->rhs_, [=](const Expr &e) {
                return makeRealDiv(op->lhs_, oldInvertFromStore->invert(e));
            }};
        (*this)(op->rhs_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->rhs_);
    }
}

void Derivative::visit(const Min &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_,
                   makeIfExpr(makeLE(op->lhs_, op->rhs_), it->second.mathExpr(),
                              makeIntConst(0)));
        setPartial(op->rhs_,
                   makeIfExpr(makeLT(op->rhs_, op->lhs_), it->second.mathExpr(),
                              makeIntConst(0)));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Max &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_,
                   makeIfExpr(makeGE(op->lhs_, op->rhs_), it->second.mathExpr(),
                              makeIntConst(0)));
        setPartial(op->rhs_,
                   makeIfExpr(makeGT(op->rhs_, op->lhs_), it->second.mathExpr(),
                              makeIntConst(0)));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const IfExpr &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->thenCase_, makeIfExpr(op->cond_, it->second.mathExpr(),
                                             makeIntConst(0)));
        setPartial(op->elseCase_,
                   makeIfExpr(makeLNot(op->cond_), it->second.mathExpr(),
                              makeIntConst(0)));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Sqrt &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeRealDiv(it->second.mathExpr(),
                                          makeMul(makeIntConst(2), op)));
    }

    if (invertFromStore_.has_value() && invertFromStore_->match(op)) {
        auto oldInvertFromStore = invertFromStore_;
        invertFromStore_ = {
            oldInvertFromStore->store(), op->expr_, [=](const Expr &e) {
                return makeSquare(oldInvertFromStore->invert(e));
            }};
        (*this)(op->expr_);
        invertFromStore_ = oldInvertFromStore;
    } else {
        (*this)(op->expr_);
    }
}

void Derivative::visit(const Exp &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeMul(it->second.mathExpr(), op));
    }
    // Don't invert exp to ln: It's too heavy
    BaseClass::visit(op);
}

void Derivative::visit(const Ln &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeRealDiv(it->second.mathExpr(), op->expr_));
    }
    // Don't invert ln to exp: It's too heavy
    BaseClass::visit(op);
}

void Derivative::visit(const Square &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(makeIntConst(2),
                           makeMul(it->second.mathExpr(), op->expr_)));
    }
    // Don't invert squre to sqrt: It's too heavy
    BaseClass::visit(op);
}

void Derivative::visit(const Sigmoid &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(it->second.mathExpr(),
                           makeMul(makeSub(makeIntConst(1), op), op)));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Sin &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(it->second.mathExpr(), makeCos(op->expr_)));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Cos &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(it->second.mathExpr(),
                           makeSub(makeIntConst(0), makeSin(op->expr_))));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Tan &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeRealDiv(it->second.mathExpr(),
                                          makeSquare(makeCos(op->expr_))));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Tanh &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(it->second.mathExpr(),
                           makeSub(makeIntConst(1), makeSquare(op))));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Cast &op) {
    // No cast here. We handle different types in `genGrads`.
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, it->second.mathExpr());
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Abs &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeIfExpr(makeGE(op->expr_, makeIntConst(0)),
                              it->second.mathExpr(),
                              makeSub(makeIntConst(0), it->second.mathExpr())));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Intrinsic &op) {
    throw InvalidAutoGrad(FT_MSG << "Please provide gradient of " << op
                                 << " explicitly by `UserGrad`");
}

} // namespace freetensor
