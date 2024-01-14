#include <functional>
#include <unordered_map>

#include <analyze/normalize_conditional_expr.h>
#include <container_utils.h>
#include <visitor.h>

namespace freetensor {

namespace {

Expr combineCond(const Expr &lhs, const Expr &rhs) {
    return lhs.isValid() && rhs.isValid() ? makeLAnd(lhs, rhs)
           : lhs.isValid()                ? lhs
                                          : rhs;
}

Expr combineCond(const Expr &a, const Expr &b, const Expr &c) {
    return combineCond(a, combineCond(b, c));
}

Expr combineCond(const std::vector<Expr> &exprs) {
    Expr ret;
    for (auto &&item : exprs) {
        ret = combineCond(ret, item);
    }
    return ret;
}

class NormalizeConditionalExpr : public Visitor {
    typedef std::pair<Expr /* value */, Expr /* condition, maybe null */> KV;

    std::unordered_map<Expr, std::vector<KV>> results_;

  public:
    const auto &results() const { return results_; }

  protected:
    void visitExpr(const Expr &e) override {
        Visitor::visitExpr(e);
        if (e->children().empty()) {
            results_[e] = {{e, nullptr}};
        } else if (e->isUnary()) {
            auto &&u = e.as<UnaryExprNode>();
            results_[e] =
                results_.at(u->expr_) |
                views::transform([&](const KV &kv) -> KV {
                    return {makeUnary(u->nodeType(), kv.first, u->debugBlame()),
                            kv.second};
                }) |
                ranges::to<std::vector>();
        } else if (e->isBinary()) {
            auto &&b = e.as<BinaryExprNode>();
            results_[e] =
                views::cartesian_product(results_.at(b->lhs_),
                                         results_.at(b->rhs_)) |
                views::transform([&](const std::tuple<KV, KV> &t) -> KV {
                    return {makeBinary(b->nodeType(), std::get<0>(t).first,
                                       std::get<1>(t).first, b->debugBlame()),
                            combineCond(std::get<0>(t).second,
                                        std::get<1>(t).second)};
                }) |
                ranges::to<std::vector>();
        }
        ASSERT(results_.count(e));
    }

    void visit(const Load &e) override {
        Visitor::visit(e);
        std::vector<KV> result;
        std::vector<Expr> indices, conds;
        std::function<void()> recurse = [&]() {
            if (indices.size() == e->indices_.size()) {
                result.emplace_back(
                    makeLoad(e->var_, indices, e->loadType_, e->debugBlame()),
                    combineCond(conds));
            } else {
                for (const KV &kv : results_.at(e->indices_[indices.size()])) {
                    indices.emplace_back(kv.first);
                    conds.emplace_back(kv.second);
                    recurse();
                    indices.pop_back();
                    conds.pop_back();
                }
            }
        };
        recurse();
        results_[e] = std::move(result);
    }

    void visit(const LoadAtVersion &e) override {
        Visitor::visit(e);
        std::vector<KV> result;
        std::vector<Expr> indices, conds;
        std::function<void()> recurse = [&]() {
            if (indices.size() == e->indices_.size()) {
                result.emplace_back(makeLoadAtVersion(e->tapeName_, indices,
                                                      e->loadType_,
                                                      e->debugBlame()),
                                    combineCond(conds));
            } else {
                for (const KV &kv : results_.at(e->indices_[indices.size()])) {
                    indices.emplace_back(kv.first);
                    conds.emplace_back(kv.second);
                    recurse();
                    indices.pop_back();
                    conds.pop_back();
                }
            }
        };
        recurse();
        results_[e] = std::move(result);
    }

    void visit(const Cast &e) override {
        Visitor::visit(e);
        results_[e] =
            results_.at(e->expr_) | views::transform([&](const KV &kv) -> KV {
                return {makeCast(kv.first, e->destType_, e->debugBlame()),
                        kv.second};
            }) |
            ranges::to<std::vector>();
    }

    void visit(const Intrinsic &e) override {
        Visitor::visit(e);
        std::vector<KV> result;
        std::vector<Expr> params, conds;
        std::function<void()> recurse = [&]() {
            if (params.size() == e->params_.size()) {
                result.emplace_back(
                    makeIntrinsic(e->format_, params, e->retType_,
                                  e->hasSideEffect_, e->debugBlame()),
                    combineCond(conds));
            } else {
                for (const KV &kv : results_.at(e->params_[params.size()])) {
                    params.emplace_back(kv.first);
                    conds.emplace_back(kv.second);
                    recurse();
                    params.pop_back();
                    conds.pop_back();
                }
            }
        };
        recurse();
        results_[e] = std::move(result);
    }

    void visit(const IfExpr &e) override {
        Visitor::visit(e);
        auto thenCase =
            views::cartesian_product(results_.at(e->thenCase_),
                                     results_.at(e->cond_)) |
            views::transform([&](const std::tuple<KV, KV> &t) -> KV {
                return {std::get<0>(t).first,
                        combineCond(std::get<0>(t).second, std::get<1>(t).first,
                                    std::get<1>(t).second)};
            });
        auto elseCase =
            views::cartesian_product(results_.at(e->elseCase_),
                                     results_.at(e->cond_)) |
            views::transform([&](const std::tuple<KV, KV> &t) -> KV {
                return {std::get<0>(t).first,
                        combineCond(std::get<0>(t).second,
                                    makeLNot(std::get<1>(t).first),
                                    std::get<1>(t).second)};
            });
        results_[e] =
            views::concat(thenCase, elseCase) | ranges::to<std::vector>();
    }
};

} // Anonymous namespace

std::vector<std::pair<Expr /* value */, Expr /* condition, maybe null */>>
normalizeConditionalExpr(const Expr &expr) {
    NormalizeConditionalExpr visitor;
    visitor(expr);
    auto &&ret = visitor.results().at(expr);
    ASSERT(!ret.empty());
    return ret;
}

} // namespace freetensor
