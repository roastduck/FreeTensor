#include <autograd/derivative.h>
#include <autograd/replace_by_saved.h>
#include <hash.h>

namespace freetensor {

namespace {

class HashMatcher : public Visitor {
    Expr pattern_;
    bool found_ = false;

  public:
    HashMatcher(const Expr &pattern) : pattern_(pattern) {}

    bool found() const { return found_; }

  protected:
    void visitExpr(const Expr &expr) override {
        if (HashComparator{}(pattern_, expr)) {
            found_ = true;
        }
        if (found_) {
            return;
        }
        Visitor::visitExpr(expr);
    }
};

class AllReadsExcludingPattern : public Visitor {
    Expr pattern_;
    std::unordered_set<std::string> reads_;

  public:
    AllReadsExcludingPattern(const Expr &pattern) : pattern_(pattern) {}

    const auto &reads() const { return reads_; }

  protected:
    void visitExpr(const Expr &expr) override {
        if (!HashComparator{}(pattern_, expr)) {
            Visitor::visitExpr(expr);
        }
    }

    void visit(const Load &op) override {
        Visitor::visit(op);
        reads_.insert(op->var_);
    }
};

} // Anonymous namespace

bool Derivative::LazyPartialDerivative::usingStore() {
    if (!usingStore_.has_value()) {
        if (rootStore_.isValid()) {
            HashMatcher matcher{rootStore_->expr_};
            matcher(mathExpr_);
            usingStore_ = matcher.found();
        } else {
            usingStore_ = false;
        }
    }
    return *usingStore_;
}

const std::unordered_set<std::string> &
Derivative::LazyPartialDerivative::reads() {
    if (!reads_.has_value()) {
        AllReadsExcludingPattern visitor{
            rootStore_.isValid() ? (Expr)rootStore_->expr_ : nullptr};
        visitor(mathExpr_);
        reads_ = visitor.reads();
    }
    return *reads_;
}

Expr Derivative::LazyPartialDerivative::replaceExpr(
    const std::unordered_map<ID, std::string> &intermediatesMap,
    const std::unordered_map<StmtOrExprID, Expr> &versions, const Expr &expr) {
    return ReplaceBySaved{symbolTable_, intermediatesMap, versions, rootStmtID_,
                          rootStore_}
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
    for (auto &&[load, derivativeLazy] : partials_) {
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
        if (rootStore_.isValid()) {
            partials_[expr] = LazyPartialDerivative{
                symbolTableSnapshot(), partial, rootExpr_.stmtId(), rootStore_};
        } else {
            partials_[expr] = LazyPartialDerivative{
                symbolTableSnapshot(), partial, rootExpr_.stmtId()};
        }
    }
}

void Derivative::visitExpr(const Expr &expr) {
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
    rootStore_ = op;
    BaseClass::visit(op);
    rootStore_ = nullptr;
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
    BaseClass::visit(op);
}

void Derivative::visit(const Sub &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, it->second.mathExpr());
        setPartial(op->rhs_, makeSub(makeIntConst(0), it->second.mathExpr()));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Mul &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, makeMul(it->second.mathExpr(), op->rhs_));
        setPartial(op->rhs_, makeMul(it->second.mathExpr(), op->lhs_));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const RealDiv &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->lhs_, makeRealDiv(it->second.mathExpr(), op->rhs_));
        setPartial(op->rhs_,
                   makeSub(makeIntConst(0),
                           makeRealDiv(makeMul(it->second.mathExpr(), op->lhs_),
                                       makeMul(op->rhs_, op->rhs_))));
    }
    BaseClass::visit(op);
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
    BaseClass::visit(op);
}

void Derivative::visit(const Exp &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeMul(it->second.mathExpr(), op));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Ln &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_, makeRealDiv(it->second.mathExpr(), op->expr_));
    }
    BaseClass::visit(op);
}

void Derivative::visit(const Square &op) {
    if (auto it = partials_.find(op); it != partials_.end()) {
        setPartial(op->expr_,
                   makeMul(makeIntConst(2),
                           makeMul(it->second.mathExpr(), op->expr_)));
    }
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
    throw InvalidAutoGrad("Please provide gradient of " + toString(op) +
                          " explicitly by `UserGrad`");
}

} // namespace freetensor
