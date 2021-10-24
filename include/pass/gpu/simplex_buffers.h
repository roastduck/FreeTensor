#ifndef GPU_SIMPLEX_BUFFERS_H
#define GPU_SIMPLEX_BUFFERS_H

#include <unordered_set>

#include <analyze/analyze_linear.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

namespace gpu {

struct SimplexOffset {
    std::unordered_map<std::string, int> offset_; // parallel scope -> offset
};

class FindSimplexOffset : public Visitor {
    std::unordered_map<std::string, std::vector<Ref<SimplexOffset>>>
        offsets_; // def ID -> [offset for each index]
    std::unordered_map<std::string, VarDef> defs_;
    std::unordered_map<std::string, std::string> var2para_;
    AnalyzeLinear analyzeLinear_;

  public:
    const std::unordered_map<std::string, std::vector<Ref<SimplexOffset>>> &
    offsets() const {
        return offsets_;
    }

  private:
    Ref<SimplexOffset>
    getSimplexOffset(const std::unordered_set<std::string> &filter,
                     const Expr &expr) {
        Ref<SimplexOffset> ret = Ref<SimplexOffset>::make();
        analyzeLinear_(expr);
        for (auto &&[h, s] : analyzeLinear_.result().at(expr).coeff_) {
            if (s.a_->nodeType() == ASTNodeType::Var) {
                auto var = s.a_.as<VarNode>();
                if (var2para_.count(var->name_) &&
                    filter.count(var2para_.at(var->name_))) {
                    ASSERT(!ret->offset_.count(var2para_.at(var->name_)));
                    ret->offset_[var2para_.at(var->name_)] = s.k_;
                }
            }
        }
        return ret;
    }

    template <class T> void visitMemAcc(const T &op) {
        Visitor::visit(op);

        auto mtype = defs_.at(op->var_)->buffer_->mtype();
        if (mtype != MemType::GPULocal && mtype != MemType::GPUShared) {
            return;
        }

        auto &&defId = defs_.at(op->var_)->id();
        std::vector<Ref<SimplexOffset>> thisOffsets;
        for (auto &&idx : op->indices_) {
            Ref<SimplexOffset> offset;
            if (mtype == MemType::GPUShared) {
                offset = getSimplexOffset(
                    {"blockIdx.x", "blockIdx.y", "blockIdx.z"}, idx);
            } else {
                offset = getSimplexOffset({"threadIdx.x", "threadIdx.y",
                                           "threadIdx.z", "blockIdx.x",
                                           "blockIdx.y", "blockIdx.z"},
                                          idx);
            }
            thisOffsets.emplace_back(offset);
        }

        if (!offsets_.count(defId)) {
            offsets_[defId] = thisOffsets;
        } else {
            ASSERT(offsets_.at(defId).size() == thisOffsets.size());
            for (size_t i = 0, n = thisOffsets.size(); i < n; i++) {
                auto &&old = offsets_.at(defId)[i];
                auto &&cur = thisOffsets[i];
                if (old.isValid() &&
                    (!cur.isValid() || old->offset_ != cur->offset_)) {
                    offsets_.at(defId)[i] = nullptr;
                }
            }
        }
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
    void visit(const Load &op) override { visitMemAcc(op); }
    void visit(const Store &op) override { visitMemAcc(op); }
    void visit(const ReduceTo &op) override { visitMemAcc(op); }
};

class ApplySimplexOffset : public Mutator {
    const std::unordered_map<std::string, std::vector<Ref<SimplexOffset>>>
        &offsets_; // def ID -> [offset for each index]
    std::unordered_map<std::string, VarDef> defs_;
    std::unordered_map<std::string, std::string> para2var_;

  public:
    ApplySimplexOffset(
        const std::unordered_map<std::string, std::vector<Ref<SimplexOffset>>>
            &offsets)
        : offsets_(offsets) {}

  private:
    template <class T> T visitMemAcc(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();

        auto &&defId = defs_.at(op->var_)->id();
        if (offsets_.count(defId)) {
            auto &&offset = offsets_.at(defId);
            ASSERT(offset.size() == op->indices_.size());
            for (size_t i = 0, n = offset.size(); i < n; i++) {
                if (offset[i].isValid()) {
                    for (auto &&[scope, k] : offset[i]->offset_) {
                        op->indices_[i] =
                            makeSub(op->indices_[i],
                                    makeMul(makeIntConst(k),
                                            makeVar(para2var_.at(scope))));
                    }
                }
            }
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Expr visit(const Load &op) override { return visitMemAcc(op); }
    Stmt visit(const Store &op) override { return visitMemAcc(op); }
    Stmt visit(const ReduceTo &op) override { return visitMemAcc(op); }
};

/**
 * If a shared or local VarDef is outside a parallel For region, it can be
 * shrinked so that each thread or block will access the same indices for
 * different data
 *
 * E.g. Alter from `local[threadIdx.x, i]` to `local[i]`
 */
Stmt simplexBuffers(const Stmt &op);

DEFINE_PASS_FOR_FUNC(simplexBuffers)

} // namespace gpu

} // namespace ir

#endif // GPU_SIMPLEX_BUFFERS_H
