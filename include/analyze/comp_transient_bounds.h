#ifndef COMP_TRANSIENT_BOUNDS_H
#define COMP_TRANSIENT_BOUNDS_H

#include <unordered_set>

#include <analyze/all_uses.h>
#include <analyze/analyze_linear.h>
#include <analyze/as_dnf.h>
#include <container_utils.h>
#include <hash.h>
#include <math/bounds.h>
#include <maybe_void.h>
#include <stmt.h>

namespace ir {

struct TransientBound {
    Expr expr_;
    std::vector<Expr> lower_, upper_;
};

class CompTransientBoundsInterface {
  public:
    virtual TransientBound transient(const Expr &op) const = 0;
    virtual const std::vector<Expr> &conds() const = 0;
};

/**
 * Compute bounds of IDENTICAL INTEGER (sub)expressions AT A POSITION in the AST
 *
 * E.g.
 *
 * ```
 * if (x <= 2) {
 *   ... = x + x; // <- AT THIS POSITION
 * }
 * ```
 *
 * At the position above, ALL TWO IDENTICAL EXPRESSIONS `x` have an upper bound
 * 2
 *
 * Invoke pass/annotate_conds before this analysis to get better accuracy
 *
 * Inherit this pass to use it. The BaseClass should be a decent of
 * WithTypeInfer<SymbolTable<...>>
 */
template <class BaseClass>
class CompTransientBounds : public BaseClass,
                            public CompTransientBoundsInterface {
    // Bounds related to certain expressions
    // Bounds in transients_ has already been recursed with (*this)(...)
    ASTHashMap<Expr, TransientBound> transients_;

    // Original bounds
    std::vector<Expr> conds_;

  public:
    TransientBound transient(const Expr &op) const override {
        if (transients_.count(op)) {
            return transients_.at(op);
        }
        return {};
    }

    const std::vector<Expr> &conds() const override { return conds_; }

  private:
    void applyCond(const Expr &_cond,
                   const std::unordered_set<std::string> &bodyAllWrites) {
        auto dnf = asDNF(_cond);

        if (dnf.size() != 1) {
            return; // Currently we cannot handle OR
        }

        for (auto &&cond : dnf.front()) {
            if (hasIntersect(allReads(cond), bodyAllWrites)) {
                continue;
            }

            auto norm = linearComp(cond);
            if (!norm.isValid()) {
                continue;
            }

            auto &&[lin, type] = *norm;
            if (!isInt(this->dtype(lin2expr(lin)))) {
                continue;
            }

            for (auto &&[k, a] : lin.coeff_) {
                if (a->nodeType() == ASTNodeType::Var ||
                    a->nodeType() == ASTNodeType::Load) {
                    auto [lower, upper] = lin2bounds(lin, type, a);
                    if (lower.isValid()) {
                        transients_[a].expr_ = a;
                        transients_[a].lower_.emplace_back(lower->expr());
                    }
                    if (upper.isValid()) {
                        transients_[a].expr_ = a;
                        transients_[a].upper_.emplace_back(upper->expr());
                    }
                }
            }

            // A temporary hack for
            // test/2.pass/test_shrink_var.py::test_const_in_branch (FIXME)
            if (type == ASTNodeType::EQ && lin.coeff_.size() == 1) {
                auto &&[k, a] = lin.coeff_.front();
                auto b = makeIntConst(-lin.bias_ / k);
                transients_[b].expr_ = b;
                transients_[b].lower_.emplace_back(a);
                transients_[b].upper_.emplace_back(a);
            }

            conds_.emplace_back(cond);
        }
    }

  protected:
    using BaseClass::visit; // Avoid hiding virtual functions

    typename BaseClass::StmtRetType visit(const For &op) override {
        MAYBE_VOID(begin, (*this)(op->begin_));
        MAYBE_VOID(end, (*this)(op->end_));
        MAYBE_VOID(step, (*this)(op->step_));
        MAYBE_VOID(len, (*this)(op->len_));

        auto var = makeVar(op->iter_);
        if (transients_.count(var)) {
            throw InvalidProgram(
                "iterators with the same name in nested loops are not allowed");
        }
        auto oldCondsSize = conds_.size();
        if (op->step_->nodeType() == ASTNodeType::IntConst) {
            auto step = op->step_.as<IntConstNode>()->val_;
            if (step > 0) {
                transients_[var] = {
                    var, {op->begin_}, {makeSub(op->end_, makeIntConst(1))}};
                conds_.emplace_back(makeGE(var, op->begin_));
                conds_.emplace_back(makeLT(var, op->end_));
                conds_.emplace_back(
                    makeEQ(makeMod(makeSub(var, op->begin_), op->step_),
                           makeIntConst(0)));
            } else if (step < 0) {
                transients_[var] = {
                    var, {makeAdd(op->end_, makeIntConst(1))}, {op->begin_}};
                conds_.emplace_back(makeLE(var, op->begin_));
                conds_.emplace_back(makeGT(var, op->end_));
                // ISL does not support negative divisor
                conds_.emplace_back(
                    makeEQ(makeMod(makeSub(op->begin_, var),
                                   makeSub(makeIntConst(0), op->step_)),
                           makeIntConst(0)));
            } else {
                transients_[var] = {var, {op->begin_}, {op->begin_}};
                conds_.emplace_back(makeEQ(var, op->begin_));
            }
        }
        this->pushFor(op);
        MAYBE_VOID(body, (*this)(op->body_));
        this->popFor(op);
        conds_.resize(oldCondsSize);
        transients_.erase(var);

        if constexpr (!std::is_same_v<typename BaseClass::StmtRetType, void>) {
            auto ret = makeFor(op->id(), op->iter_, std::move(begin),
                               std::move(end), std::move(step), std::move(len),
                               op->property_, std::move(body));
            return COPY_DEBUG_INFO(ret, op);
        }
    }

    typename BaseClass::StmtRetType visit(const If &op) override {
        MAYBE_VOID(cond, (*this)(op->cond_));

        auto oldMap = transients_;
        auto oldCondsSize = conds_.size();
        applyCond(op->cond_, allWrites(op->thenCase_));
        MAYBE_VOID(thenCase, (*this)(op->thenCase_));
        transients_ = oldMap;
        conds_.resize(oldCondsSize);

        [[maybe_unused]] Stmt elseCase = nullptr;
        if (op->elseCase_.isValid()) {
            auto oldCondsSize = conds_.size();
            applyCond(makeLNot(op->cond_), allWrites(op->elseCase_));
            MAYBE_VOID_ASSIGN(elseCase, (*this)(op->elseCase_));
            transients_ = oldMap;
            conds_.resize(oldCondsSize);
        }

        if constexpr (!std::is_same_v<typename BaseClass::StmtRetType, void>) {
            auto ret = makeIf(op->id(), std::move(cond), std::move(thenCase),
                              std::move(elseCase));
            return COPY_DEBUG_INFO(ret, op);
        }
    }

    typename BaseClass::StmtRetType visit(const Assert &op) override {
        MAYBE_VOID(cond, (*this)(op->cond_));

        auto oldMap = transients_;
        auto oldCondsSize = conds_.size();
        applyCond(op->cond_, allWrites(op->body_));
        MAYBE_VOID(body, (*this)(op->body_));
        transients_ = oldMap;
        conds_.resize(oldCondsSize);

        if constexpr (!std::is_same_v<typename BaseClass::StmtRetType, void>) {
            auto ret = makeAssert(op->id(), std::move(cond), std::move(body));
            return COPY_DEBUG_INFO(ret, op);
        }
    }

    typename BaseClass::StmtRetType visit(const Assume &op) override {
        MAYBE_VOID(cond, (*this)(op->cond_));

        auto oldMap = transients_;
        auto oldCondsSize = conds_.size();
        applyCond(op->cond_, allWrites(op->body_));
        MAYBE_VOID(body, (*this)(op->body_));
        transients_ = oldMap;
        conds_.resize(oldCondsSize);

        if constexpr (!std::is_same_v<typename BaseClass::StmtRetType, void>) {
            auto ret = makeAssume(op->id(), std::move(cond), std::move(body));
            return COPY_DEBUG_INFO(ret, op);
        }
    }
};

} // namespace ir

#endif // COMP_TRANSIENT_BOUNDS_H
