#ifndef COUNT_CONTIG_ACCESS_LOOPS_H
#define COUNT_CONTIG_ACCESS_LOOPS_H

#include <unordered_map>

#include <analyze/analyze_linear.h>
#include <visitor.h>

namespace ir {

class CountContigAccessLoops : public Visitor {
    std::unordered_map<std::string, int> counts_;
    std::unordered_map<std::string, For> var2for_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    AnalyzeLinear analyzeLinear_;

  public:
    const std::unordered_map<std::string, int> &counts() const {
        return counts_;
    }

  private:
    template <class T> void visitMemAccess(const T &op) {
        Visitor::visit(op);
        if (buffers_.at(op->var_)->atype() == AccessType::Cache) {
            // We don't count Cache vars here because they are likely
            // registers
            return;
        }
        if (!op->indices_.empty()) {
            auto idx = op->indices_.back();
            analyzeLinear_(idx);
            for (auto &&[h, s] : analyzeLinear_.result().at(idx).coeff_) {
                if (s.k_ == 1 && s.a_->nodeType() == ASTNodeType::Var) {
                    auto &&var = s.a_.template as<VarNode>();
                    counts_[var2for_.at(var->name_)->id()]++;
                }
            }
        }
    }

  protected:
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
    void visit(const Load &op) override { visitMemAccess(op); }
    void visit(const Store &op) override { visitMemAccess(op); }
    void visit(const ReduceTo &op) override { visitMemAccess(op); }
};

} // namespace ir

#endif // COUNT_CONTIG_ACCESS_LOOPS_H
