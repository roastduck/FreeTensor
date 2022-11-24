#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <func.h>
#include <memory>
#include <mutator.h>

namespace freetensor {

namespace gpu {

class NormalizeVarInKernel : public CompTransientBounds<SymbolTable<Mutator>> {
    typedef CompTransientBounds<SymbolTable<Mutator>> BaseClass;

    std::unordered_set<std::string> legalNames_;

    std::vector<VarDef> varsToHoist_;
    bool inKernel_ = false;

    std::unique_ptr<CompUniqueBounds> uniqueOwn_;
    CompUniqueBounds &unique_;

  public:
    NormalizeVarInKernel()
        : uniqueOwn_(std::make_unique<CompUniqueBoundsCombination>(*this)),
          unique_(*uniqueOwn_) {}

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
};

/**
 * The shape of all variables defined inside a kernel must be determined outside
 * the kernel
 */
Stmt normalizeVarInKernel(const Stmt &s);

DEFINE_PASS_FOR_FUNC(normalizeVarInKernel)

} // namespace gpu

} // namespace freetensor
