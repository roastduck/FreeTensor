#ifndef FREE_TENSOR_HOIST_RETURN_VARS_H
#define FREE_TENSOR_HOIST_RETURN_VARS_H

#include <unordered_map>
#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace freetensor {

/**
 * Hoist a returning VarDef out of any loops
 *
 * A VarDef can be marked as a returning value in a Func. These variables are
 * allocated at run time. However, directly marking a VarDef as a returning
 * value may leave it inside a loop, but a return statement inside a loop does
 * not make sense. It will cause more problems if we parallelize the loop. This
 * pass solves the problem
 */
class HoistReturnVars : public Mutator {
    Func func_;
    std::optional<ID> outMostLoop_;
    std::vector<VarDef> toHoist_; // inner to outer

  public:
    HoistReturnVars(const Func &func) : func_(func) {}

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
};

Func hoistReturnVars(const Func &func);

} // namespace freetensor

#endif // FREE_TENSOR_HOIST_RETURN_VARS_H
