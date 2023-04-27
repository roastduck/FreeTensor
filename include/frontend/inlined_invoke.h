#ifndef FREE_TENSOR_INLINED_INVOKE_H
#define FREE_TENSOR_INLINED_INVOKE_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <frontend/frontend_var.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class StripReturns : public Mutator {
    const std::vector<FuncRet> &returns_;
    std::vector<Ref<Buffer>> bufToReturn_; // from outer to inner

  public:
    StripReturns(const std::vector<FuncRet> &returns) : returns_(returns) {}

    const std::vector<Ref<Buffer>> &bufToReturn() const { return bufToReturn_; }

  protected:
    Stmt visit(const VarDef &op) override;
};

class InlinedInvoke : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    Metadata callSiteMetadata_;
    const std::unordered_map<std::string, Ref<FrontendVar>> &kvs_;
    const std::unordered_map<std::string, std::string> &renameRets_;
    const std::unordered_set<std::string> &conflictNames_;

  public:
    InlinedInvoke(
        const Metadata &callSiteMetadata,
        const std::unordered_map<std::string, Ref<FrontendVar>> &kvs,
        const std::unordered_map<std::string, std::string> &renameRets,
        const std::unordered_set<std::string> &conflictNames)
        : callSiteMetadata_(callSiteMetadata), kvs_(kvs),
          renameRets_(renameRets), conflictNames_(conflictNames) {
        ASSERT(callSiteMetadata_.isValid());
    }

  protected:
    using BaseClass::visit;
    Stmt visitStmt(const Stmt &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

/**
 * A preprocessing step between `inlinedInvoke`
 *
 * If the callee returns some tensors, gather information of them, in an
 * outer-to-inner order. The caller then needs to create scopes for these
 * tensors, before finally calling `inlinedInvoke` inside the scopes
 */
std::pair<Func, std::vector<Ref<Buffer>>> stripReturns(const Func &func);

/**
 * Replace a Function's all arguments by `FrontendVar`s and return a `Stmt`
 *
 * Usually we handle function calls directly in the frontend. But sometimes we
 * may want to call a differentiated function, which is already lowered as an
 * AST. Then, we can use `inlinedInvoke` to call it
 */
Stmt inlinedInvoke(const Metadata &callSiteMetadata, const Func &func,
                   const std::vector<Ref<FrontendVar>> &args,
                   const std::unordered_map<std::string, Ref<FrontendVar>> &kvs,
                   const std::vector<std::string> &retNames,
                   const std::unordered_set<std::string> &conflictNames);

} // namespace freetensor

#endif // FREE_TENSOR_INLINED_INVOKE_H
