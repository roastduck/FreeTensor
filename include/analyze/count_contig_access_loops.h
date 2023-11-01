#ifndef FREE_TENSOR_COUNT_CONTIG_ACCESS_LOOPS_H
#define FREE_TENSOR_COUNT_CONTIG_ACCESS_LOOPS_H

#include <unordered_map>

#include <analyze/analyze_linear.h>
#include <analyze/symbol_table.h>
#include <math/utils.h>
#include <pass/const_fold.h>
#include <visitor.h>

namespace freetensor {

class CountContigAccessLoops : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::unordered_map<ID, std::pair<int64_t, int>>
        counts_; // for ID -> (count, -depth)
    AnalyzeLinear analyzeLinear_;
    int64_t repeat_ = 1;
    int depth_ = 0;

  public:
    const std::unordered_map<ID, std::pair<int64_t, int>> &counts() const {
        return counts_;
    }

  private:
    int64_t getStaticSize(const std::string &var) {
        int64_t ret = 1;
        for (auto &&dim : buffer(var)->tensor()->shape()) {
            if (dim->nodeType() == ASTNodeType::IntConst) {
                ret *= dim.as<IntConstNode>()->val_;
            } else {
                return -1;
            }
        }
        return ret;
    }

    void countContigVars(std::unordered_map<std::string, int> *cnt,
                         const Expr &expr, const Expr &modP = nullptr) {
        analyzeLinear_(expr);
        for (auto &&[_k, a] : analyzeLinear_.result().at(expr).coeff_) {
            int64_t k = _k;
            if (modP.isValid()) {
                // TODO: Dynamic p: (p - 1) === -1, mod p
                if (auto _p = constFold(modP);
                    _p->nodeType() == ASTNodeType::IntConst) {
                    auto p = _p.as<IntConstNode>()->val_;
                    if (mod(k, p) == mod(1, p) || mod(k, p) == mod(-1, p)) {
                        goto ok;
                    }
                }
            }
            if (k == 1 || k == -1) {
                goto ok;
            }
            continue;

        ok:
            switch (a->nodeType()) {
            case ASTNodeType::Var:
                (*cnt)[a.template as<VarNode>()->name_] += repeat_;
                break;
            case ASTNodeType::Mod:
                // TODO: ASTNodeType::Remainder
                countContigVars(cnt, a.as<BinaryExprNode>()->lhs_,
                                a.as<BinaryExprNode>()->rhs_);
                break;
            default:;
            }
        }
    }

    template <class T> void visitMemAccess(const T &op) {
        BaseClass::visit(op);
        auto size = getStaticSize(op->var_);
        if (size != -1 && size < 128) {
            // We don't count too small vars here because they are likely
            // registers
            return;
        }
        if (!op->indices_.empty()) {
            std::unordered_map<std::string, int> cnt;
            countContigVars(&cnt, op->indices_.back());
            for (auto &&[v, c] : cnt) {
                counts_[loop(v)->id()].first += c;
            }
        }
    }

  protected:
    using BaseClass::visit;
    void visit(const For &op) override;
    void visit(const Load &op) override { visitMemAccess(op); }
    void visit(const Store &op) override { visitMemAccess(op); }
    void visit(const ReduceTo &op) override { visitMemAccess(op); }
    void visit(const MatMul &op) override {} // do nothing
};

} // namespace freetensor

#endif // FREE_TENSOR_COUNT_CONTIG_ACCESS_LOOPS_H
