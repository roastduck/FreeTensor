#ifndef FREE_TENSOR_GPU_SIMPLEX_BUFFERS_H
#define FREE_TENSOR_GPU_SIMPLEX_BUFFERS_H

#ifdef FT_WITH_CUDA

#include <unordered_set>

#include <analyze/analyze_linear.h>
#include <analyze/symbol_table.h>
#include <container_utils.h>
#include <func.h>
#include <hash.h>
#include <mutator.h>
#include <pass/replace_iter.h>
#include <visitor.h>

namespace freetensor {

namespace gpu {

struct SimplexOffset {
    ASTHashSet<Expr> offset_;
};

class FindSimplexOffset : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    ID defId_;
    std::unordered_map<ID, std::vector<Ref<SimplexOffset>>>
        offsets_; // def ID -> [offset for each index]
    AnalyzeLinear analyzeLinear_;

  public:
    FindSimplexOffset(const ID &defId = ID()) : defId_(defId) {}

    const std::unordered_map<ID, std::vector<Ref<SimplexOffset>>> &
    offsets() const {
        return offsets_;
    }

  private:
    Ref<SimplexOffset>
    getSimplexOffset(const std::unordered_set<ParallelScope> &filter,
                     const Expr &expr);

    template <class T> void visitMemAcc(const T &op) {
        BaseClass::visit(op);

        auto mtype = buffer(op->var_)->mtype();
        if (mtype != MemType::GPULocal && mtype != MemType::GPUShared &&
            mtype != MemType::GPUWarp) {
            return;
        }

        auto &&defId = def(op->var_)->id();
        if (defId_.isValid() && defId_ != defId) {
            return;
        }

        std::vector<Ref<SimplexOffset>> thisOffsets;
        for (auto &&idx : op->indices_) {
            Ref<SimplexOffset> offset;
            if (mtype == MemType::GPUShared || mtype == MemType::GPUWarp) {
                offset =
                    getSimplexOffset({blockIdxX, blockIdxY, blockIdxZ}, idx);
            } else {
                offset = getSimplexOffset({threadIdxX, threadIdxY, threadIdxZ,
                                           blockIdxX, blockIdxY, blockIdxZ},
                                          idx);
            }
            thisOffsets.emplace_back(offset);
        }

        if (!offsets_.count(defId)) {
            offsets_[defId] = thisOffsets;
        } else {
            ASSERT(offsets_.at(defId).size() == thisOffsets.size());
            for (auto &&[old, cur] :
                 views::zip(offsets_.at(defId), thisOffsets)) {
                if (old.isValid() &&
                    (!cur.isValid() || old->offset_ != cur->offset_)) {
                    old = nullptr;
                }
            }
        }
    }

  protected:
    using BaseClass::visit;
    void visit(const Load &op) override { visitMemAcc(op); }
    void visit(const Store &op) override { visitMemAcc(op); }
    void visit(const ReduceTo &op) override { visitMemAcc(op); }
};

class ApplySimplexOffset : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, std::vector<Ref<SimplexOffset>>>
        &offsets_; // def ID -> [offset for each index]
    std::unordered_map<std::string, Expr> para2var_;

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
            for (auto &&[off, idx] : views::zip(offset, op->indices_)) {
                if (off.isValid()) {
                    for (auto &&expr : off->offset_) {
                        idx = makeSub(idx, ReplaceIter{para2var_}(expr));
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
 *
 * @param defId : If set, only alter this VarDef
 */
Stmt simplexBuffers(const Stmt &op, const ID &defId = ID());

DEFINE_PASS_FOR_FUNC(simplexBuffers)

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_SIMPLEX_BUFFERS_H
