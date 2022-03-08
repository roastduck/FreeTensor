#ifndef GPU_SIMPLEX_BUFFERS_H
#define GPU_SIMPLEX_BUFFERS_H

#include <unordered_set>

#include <itertools.hpp>

#include <analyze/analyze_linear.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

namespace gpu {

struct SimplexOffset {
    std::unordered_map<ID, int> offset_; // parallel scope -> offset
};

class FindSimplexOffset : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::unordered_map<ID, std::vector<Ref<SimplexOffset>>>
        offsets_; // def ID -> [offset for each index]
    std::unordered_map<std::string, ID> var2para_;
    AnalyzeLinear analyzeLinear_;

  public:
    const std::unordered_map<ID, std::vector<Ref<SimplexOffset>>> &
    offsets() const {
        return offsets_;
    }

  private:
    Ref<SimplexOffset> getSimplexOffset(const std::unordered_set<ID> &filter,
                                        const Expr &expr) {
        Ref<SimplexOffset> ret = Ref<SimplexOffset>::make();
        analyzeLinear_(expr);
        for (auto &&[k, a] : analyzeLinear_.result().at(expr).coeff_) {
            if (a->nodeType() == ASTNodeType::Var) {
                auto var = a.as<VarNode>();
                if (var2para_.count(var->name_) &&
                    filter.count(var2para_.at(var->name_))) {
                    ASSERT(!ret->offset_.count(var2para_.at(var->name_)));
                    ret->offset_[var2para_.at(var->name_)] = k;
                }
            }
        }
        return ret;
    }

    template <class T> void visitMemAcc(const T &op) {
        BaseClass::visit(op);

        auto mtype = buffer(op->var_)->mtype();
        if (mtype != MemType::GPULocal && mtype != MemType::GPUShared && mtype != MemType::GPUWarp) {
            return;
        }

        auto &&defId = def(op->var_)->id();
        std::vector<Ref<SimplexOffset>> thisOffsets;
        for (auto &&idx : op->indices_) {
            Ref<SimplexOffset> offset;
            if (mtype == MemType::GPUShared || mtype == MemType::GPUWarp) {
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
            for (auto &&[old, cur] :
                 iter::zip(offsets_.at(defId), thisOffsets)) {
                if (old.isValid() &&
                    (!cur.isValid() || old->offset_ != cur->offset_)) {
                    old = nullptr;
                }
            }
        }
    }

  protected:
    using BaseClass::visit;
    void visit(const For &op) override;
    void visit(const Load &op) override { visitMemAcc(op); }
    void visit(const Store &op) override { visitMemAcc(op); }
    void visit(const ReduceTo &op) override { visitMemAcc(op); }
};

class ApplySimplexOffset : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, std::vector<Ref<SimplexOffset>>>
        &offsets_; // def ID -> [offset for each index]
    std::unordered_map<ID, std::string> para2var_;

  public:
    ApplySimplexOffset(
        const std::unordered_map<ID, std::vector<Ref<SimplexOffset>>> &offsets)
        : offsets_(offsets) {}

  private:
    template <class T> T visitMemAcc(const T &_op) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();

        auto &&defId = def(op->var_)->id();
        if (offsets_.count(defId)) {
            auto &&offset = offsets_.at(defId);
            ASSERT(offset.size() == op->indices_.size());
            for (auto &&[off, idx] : iter::zip(offset, op->indices_)) {
                if (off.isValid()) {
                    for (auto &&[scope, k] : off->offset_) {
                        idx =
                            makeSub(idx, makeMul(makeIntConst(k),
                                                 makeVar(para2var_.at(scope))));
                    }
                }
            }
        }
        return op;
    }

  protected:
    using BaseClass::visit;
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
