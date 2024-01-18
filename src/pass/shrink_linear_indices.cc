#include <algorithm>
#include <unordered_map>

#include <analyze/all_defs.h>
#include <analyze/analyze_linear.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds_combination.h>
#include <container_utils.h>
#include <math/utils.h>
#include <mutator.h>
#include <pass/shrink_linear_indices.h>
#include <visitor.h>

namespace freetensor {

namespace {

struct IntBound {
    int64_t lower_, upper_;
};

class GatherLinearIndices : public CompTransientBounds<Visitor> {
    typedef CompTransientBounds<Visitor> BaseClass;

    ID vardef_;
    std::string var_;

    std::vector<std::unordered_map<int64_t /* coeff */, IntBound>> bounds_;

    Ref<CompUniqueBounds> unique_;

  public:
    GatherLinearIndices(const ID &vardef) : vardef_(vardef) {}

    const auto &bounds() const { return bounds_; }

  private:
    template <typename T> void visitAcc(const T &op) {
        BaseClass::visit(op);
        if (op->var_ == var_) {
            ASSERT(bounds_.size() == op->indices_.size());
            for (auto &&[idx, bound] : views::zip(op->indices_, bounds_)) {
                auto lin = linear(idx);
                for (auto &&[_k, a] : lin.coeff_) {
                    int k = _k;
                    auto l = unique_->getIntLower(a);
                    auto u = unique_->getIntUpper(a);
                    if (k < 0) {
                        k = -k;
                        l = -l;
                        u = -u;
                        std::swap(l, u);
                    }
                    if (!bound.count(k)) {
                        bound[k] = {l, u};
                    } else {
                        bound[k].lower_ = std::min(bound[k].lower_, l);
                        bound[k].upper_ = std::max(bound[k].upper_, u);
                    }
                }
            }
        }
    }

  protected:
    using BaseClass::visit;

    void visitStmt(const Stmt &s) override {
        // CompUniqueBounds requires one instance per Stmt
        auto uniqueOfOuterStmt = unique_;
        unique_ = Ref<CompUniqueBoundsCombination>::make(*this);
        BaseClass::visitStmt(s);
        unique_ = uniqueOfOuterStmt;
    }

    void visit(const VarDef &op) override {
        if (op->id() == vardef_) {
            var_ = op->name_;
            bounds_.resize(op->buffer_->tensor()->shape().size());
            BaseClass::visit(op);
            var_.clear();
        } else {
            BaseClass::visit(op);
        }
    }

    void visit(const Load &op) override { visitAcc(op); }
    void visit(const Store &op) override { visitAcc(op); }
    void visit(const ReduceTo &op) override { visitAcc(op); }
};

class ReplaceLinearIndices : public Mutator {
    ID vardef_;
    std::string var_;

    const std::vector<std::unordered_map<int64_t, int64_t>> &replace_;

  public:
    ReplaceLinearIndices(
        const ID &vardef,
        const std::vector<std::unordered_map<int64_t, int64_t>> &replace)
        : vardef_(vardef), replace_(replace) {}

  private:
    template <typename T> auto visitAcc(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (op->var_ == var_) {
            for (auto &&[idx, rep] : views::zip(op->indices_, replace_)) {
                auto lin = linear(idx);
                for (auto &[k, a] : lin.coeff_) {
                    k = rep.at(k);
                }
                idx = lin2expr(lin);
            }
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override {
        if (op->id() == vardef_) {
            var_ = op->name_;
            auto ret = Mutator::visit(op);
            var_.clear();
            return ret;
        } else {
            return Mutator::visit(op);
        }
    }

    Expr visit(const Load &op) override { return visitAcc(op); }
    Stmt visit(const Store &op) override { return visitAcc(op); }
    Stmt visit(const ReduceTo &op) override { return visitAcc(op); }
};

} // Anonymous namespace

Stmt shrinkLinearIndices(const Stmt &_ast, const ID &vardef) {
    Stmt ast = _ast;

    GatherLinearIndices gather{vardef};
    gather(ast);
    auto &&bounds = gather.bounds();

    bool needMutation = false;
    std::vector<std::unordered_map<int64_t, int64_t>> replaceCoeff;
    for (auto &&_bound : bounds) {
        auto bound =
            _bound | ranges::to<std::vector<std::pair<int64_t, IntBound>>>();
        std::sort(bound.begin(), bound.end(),
                  [](const auto &lhs, const auto &rhs) {
                      return lhs.first > rhs.first;
                  }); // Sort k from high to low
        std::vector<int64_t> newCoeff =
            bound | views::keys | ranges::to<std::vector>();
        for (size_t n = bound.size(), i = n - 1; ~i; i--) {
            int g = newCoeff[0];
            for (size_t j = 1; j <= i; j++) {
                g = gcd(g, newCoeff[j]);
            }
            int64_t l = LLONG_MAX, u = LLONG_MIN;
            if (i + 1 < n) {
                for (size_t j = i + 1; j < n; j++) {
                    l = std::min(l, newCoeff[j] * bound[j].second.lower_);
                    u = std::max(u, newCoeff[j] * bound[j].second.upper_);
                }
            } else {
                l = u = 0;
            }
            if (u - l + 1 < g) {
                for (size_t j = 0; j <= i; j++) {
                    newCoeff[j] = newCoeff[j] / g * (u - l + 1);
                }
                needMutation = true;
            }
        }
        replaceCoeff.emplace_back(views::zip(bound | views::keys, newCoeff) |
                                  ranges::to<std::unordered_map>());
    }

    if (needMutation) {
        ast = ReplaceLinearIndices{vardef, replaceCoeff}(ast);
    }

    return ast;
}

Stmt shrinkLinearIndices(const Stmt &_ast) {
    Stmt ast = _ast;
    for (auto &&[varDefId, name] : allDefs(ast, {AccessType::Cache})) {
        ast = shrinkLinearIndices(ast, varDefId);
    }
    return ast;
}

} // namespace freetensor
