#ifndef FREE_TENSOR_INLINED_INVOKE_H
#define FREE_TENSOR_INLINED_INVOKE_H

#include <unordered_map>

#include <frontend/frontend_var.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class InlinedInvoke : public Mutator {
    Metadata callSiteMetadata_;
    const std::unordered_map<std::string, Ref<FrontendVar>> &kvs_;

  public:
    InlinedInvoke(const Metadata &callSiteMetadata,
                  const std::unordered_map<std::string, Ref<FrontendVar>> &kvs)
        : callSiteMetadata_(callSiteMetadata), kvs_(kvs) {}

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * Replace a Function's all arguments by `FrontendVar`s and return an Stmt
 *
 * Usually we handle function calls directly in the frontend. But sometimes we
 * may want to call a differentiated function, which is already lowered as an
 * AST. Then, we can use `inlinedInvoke` to call it
 */
Stmt inlinedInvoke(
    const Metadata &callSiteMetadata, const Func &func,
    const std::vector<Ref<FrontendVar>> &args,
    const std::unordered_map<std::string, Ref<FrontendVar>> &kvs);

} // namespace freetensor

#endif // FREE_TENSOR_INLINED_INVOKE_H
