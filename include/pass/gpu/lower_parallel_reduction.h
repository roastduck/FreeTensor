#ifndef FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H
#define FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H

#ifdef FT_WITH_CUDA

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class InsertWorkspaces : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    std::unordered_map<ID, std::pair<std::string, Ref<ReductionItem>>>
        ws2red_; // workspace ID -> (loop iter name, reduction info)
    std::vector<For> loopStack_;
    std::unordered_set<std::string> handledVars_;
    bool converged_ = true;

  public:
    const auto &ws2red() const { return ws2red_; }
    bool converged() const { return converged_; }

  private:
    std::vector<std::pair<For, int>> reducedBy(const ReduceTo &op);

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class InsertBinaryReduction : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, std::pair<std::string, Ref<ReductionItem>>>
        &ws2red_; // workspace ID -> (loop iter name, reduction info)
    std::unordered_map<ID, ID>
        ws2scope_; // workspace ID -> scope that actually do the computation,
                   // excluding initialization, binary reduction and flushing

  public:
    InsertBinaryReduction(
        const std::unordered_map<ID, std::pair<std::string, Ref<ReductionItem>>>
            &ws2red)
        : ws2red_(ws2red) {}

    const auto &ws2scope() const { return ws2scope_; }

  private:
    template <class T> T visitMemAcc(const T &_op) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (auto it = ws2red_.find(def(op->var_)->id()); it != ws2red_.end()) {
            auto &&l = loop(it->second.first);
            auto nth = makeSub(makeVar(l->iter_), l->begin_);
            op->indices_.insert(op->indices_.begin(), nth);
        }
        return op;
    }

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override { return visitMemAcc(op); }
    Stmt visit(const ReduceTo &op) override { return visitMemAcc(op); }
    Expr visit(const Load &op) override { return visitMemAcc(op); }
};

class CorrectInterThreadDependence : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    const std::unordered_map<ID, std::pair<std::string, Ref<ReductionItem>>>
        &ws2red_; // workspace ID -> (loop iter name, reduction info)

    std::unordered_map<ID, std::vector<VarDef>> loop2ws_;

  public:
    CorrectInterThreadDependence(
        const std::unordered_map<ID, std::pair<std::string, Ref<ReductionItem>>>
            &ws2red)
        : ws2red_(ws2red) {}

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
};

Stmt lowerParallelReduction(const Stmt &op);

DEFINE_PASS_FOR_FUNC(lowerParallelReduction)

} // namespace gpu

} // namespace freetensor

#endif // FT_WITH_CUDA

#endif // FREE_TENSOR_GPU_LOWER_PARALLEL_REDUCTION_H
