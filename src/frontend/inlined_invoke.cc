#include <algorithm>

#include <container_utils.h>
#include <frontend/inlined_invoke.h>
#include <pass/hoist_return_vars.h>
#include <pass/undo_make_reduction.h>

namespace freetensor {

Stmt InlinedInvoke::visitStmt(const Stmt &op) {
    auto ret = Mutator::visitStmt(op);
    ret->metadata() = makeMetadata("inlined_invoke", {callSiteMetadata_});
    return ret;
}

Expr InlinedInvoke::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (kvs_.count(op->var_)) {
        std::vector<FrontendVarIdx> indices;
        indices.reserve(op->indices_.size());
        for (auto &&idx : op->indices_) {
            indices.emplace_back(FrontendVarIdx::fromSingle(idx));
        }
        auto &&fv = *kvs_.at(op->var_);
        return FrontendVar(fv.name(), fv.fullShape(), fv.dtype(), fv.mtype(),
                           fv.chainIndices(indices))
            .asLoad();
    }
    if (auto it = renameRets_.find(op->var_); it != renameRets_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt InlinedInvoke::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    if (kvs_.count(op->var_)) {
        std::vector<FrontendVarIdx> indices;
        indices.reserve(op->indices_.size());
        for (auto &&idx : op->indices_) {
            indices.emplace_back(FrontendVarIdx::fromSingle(idx));
        }
        auto &&fv = *kvs_.at(op->var_);
        return FrontendVar(fv.name(), fv.fullShape(), fv.dtype(), fv.mtype(),
                           fv.chainIndices(indices))
            .asStore(op->metadata(), op->expr_);
    }
    if (auto it = renameRets_.find(op->var_); it != renameRets_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt InlinedInvoke::visit(const ReduceTo &op) {
    ASSERT(false); // We have called undoMakeReduction
}

Stmt InlinedInvoke::visit(const VarDef &op) {
    if (kvs_.count(op->name_)) {
        if ((int)op->buffer_->tensor()->shape().size() !=
            kvs_.at(op->name_)->ndim()) {
            throw InvalidProgram("The number of dimensions of argument " +
                                 toString(*kvs_.at(op->name_)) +
                                 " does not match parameter " + op->name_);
        }
        return (*this)(op->body_);
    } else {
        return Mutator::visit(op);
    }
}

std::pair<Func, std::vector<Ref<Buffer>>> stripReturns(const Func &_func) {
    auto func = hoistReturnVars(_func);
    std::vector<Ref<Buffer>> toReturn;
    while (func->body_->nodeType() == ASTNodeType::VarDef) {
        if (auto vardef = func->body_.as<VarDefNode>();
            std::find_if(func->returns_.begin(), func->returns_.end(),
                         [&](const FuncRet &ret) {
                             return ret.name_ == vardef->name_;
                         }) != func->returns_.end()) {
            toReturn.emplace_back(vardef->buffer_);
            func->body_ = vardef->body_;
        } else {
            break;
        }
    }
    return {func, toReturn};
}

Stmt inlinedInvoke(
    const Metadata &callSiteMetadata, const Func &func,
    const std::vector<Ref<FrontendVar>> &args,
    const std::unordered_map<std::string, Ref<FrontendVar>> &_kvs,
    const std::vector<std::string> &retNames) {
    Stmt ast = func->body_;
    ast = undoMakeReduction(ast);

    auto kvs = _kvs;
    if (args.size() != func->params_.size()) {
        throw InvalidProgram(func->name_ + " has " +
                             std::to_string(func->params_.size()) +
                             " parameters, but " + std::to_string(args.size()) +
                             " arguments are provided");
    }
    for (auto &&[param, arg] : views::zip(func->params_, args)) {
        kvs[param.name_] = arg;
    }

    std::unordered_map<std::string, std::string> renameRets;
    if (retNames.size() > func->returns_.size()) {
        throw InvalidProgram(func->name_ + " has only " +
                             std::to_string(func->returns_.size()) +
                             " return values, but " +
                             std::to_string(retNames.size()) + " are required");
    }
    for (auto &&[ret, name] : views::zip(func->returns_, retNames)) {
        renameRets[ret.name_] = name;
    }

    ast = InlinedInvoke(callSiteMetadata.isValid() ? callSiteMetadata
                                                   : makeMetadata(),
                        kvs, renameRets)(ast);

    return ast;
}

} // namespace freetensor
