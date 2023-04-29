#include <algorithm>

#include <analyze/all_uses.h>
#include <container_utils.h>
#include <frontend/inlined_invoke.h>
#include <pass/hoist_return_vars.h>
#include <pass/rename_var.h>

namespace freetensor {

static std::string getNewName(const std::string &oldName,
                              const std::unordered_set<std::string> &used) {
    for (int i = 1;; i++) {
        if (auto name = oldName + "." + std::to_string(i); !used.count(name)) {
            return name;
        }
    }
}

Stmt StripReturns::visit(const VarDef &op) {
    if (std::find_if(returns_.begin(), returns_.end(), [&](const FuncRet &ret) {
            return ret.name_ == op->name_;
        }) != returns_.end()) {
        bufToReturn_.emplace_back(op->buffer_);
        return (*this)(op->body_);
    } else {
        return Mutator::visit(op);
    }
}

Stmt InlinedInvoke::visitStmt(const Stmt &op) {
    auto ret = BaseClass::visitStmt(op);
    ret->metadata() = makeMetadata("inlined_invoke", {callSiteMetadata_});
    return ret;
}

Expr InlinedInvoke::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
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
    auto __op = BaseClass::visit(_op);
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

Stmt InlinedInvoke::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    if (kvs_.count(op->var_)) {
        std::vector<FrontendVarIdx> indices;
        indices.reserve(op->indices_.size());
        for (auto &&idx : op->indices_) {
            indices.emplace_back(FrontendVarIdx::fromSingle(idx));
        }
        auto &&fv = *kvs_.at(op->var_);
        return FrontendVar(fv.name(), fv.fullShape(), fv.dtype(), fv.mtype(),
                           fv.chainIndices(indices))
            .asReduceTo(op->op_, op->metadata(), op->expr_);
    }
    if (auto it = renameRets_.find(op->var_); it != renameRets_.end()) {
        op->var_ = it->second;
    }
    return op;
}

Stmt InlinedInvoke::visit(const VarDef &op) {
    if (kvs_.count(op->name_)) {
        if ((int)op->buffer_->tensor()->shape().size() !=
            kvs_.at(op->name_)->ndim()) {
            throw InvalidProgram(
                "The number of dimensions of argument " +
                toString(*kvs_.at(op->name_)) + " (" +
                std::to_string(kvs_.at(op->name_)->ndim()) +
                ") does not match parameter " + op->name_ + " (" +
                std::to_string(op->buffer_->tensor()->shape().size()) + ")");
        }
        return (*this)(op->body_);
    } else {
        if (conflictNames_.count(op->name_)) {
            auto name =
                getNewName(op->name_, uni(conflictNames_,
                                          uni(this->names(), allNames(op))));
            auto newOp = renameVar(op, op->name_, name);
            // tail recursion, because kvs_ may contain conflict names, and we
            // don't want to rename those arguments
            return (*this)(newOp);
        } else {
            return BaseClass::visit(op);
        }
    }
}

Stmt InlinedInvoke::visit(const For &op) {
    if (conflictNames_.count(op->iter_)) {
        auto name = getNewName(
            op->iter_, uni(conflictNames_, uni(this->names(), allNames(op))));
        auto newOp = renameVar(op, op->iter_, name);
        // tail recursion, because kvs_ may contain conflict names, and we don't
        // want to rename those arguments
        return (*this)(newOp);
    } else {
        return BaseClass::visit(op);
    }
}

std::pair<Func, std::vector<Ref<Buffer>>> stripReturns(const Func &_func) {
    auto func = hoistReturnVars(_func);
    StripReturns mutator(func->returns_);
    func->body_ = mutator(func->body_);
    return {func, mutator.bufToReturn()};
}

Stmt inlinedInvoke(
    const Metadata &callSiteMetadata, const Func &func,
    const std::vector<Ref<FrontendVar>> &args,
    const std::unordered_map<std::string, Ref<FrontendVar>> &_kvs,
    const std::vector<std::string> &retNames,
    const std::unordered_set<std::string> &conflictNames,
    bool forceAllowClosures) {
    Stmt ast = func->body_;

    if (!forceAllowClosures) {
        for (auto &&param : func->params_) {
            if (param.isInClosure()) {
                throw InvalidProgram(
                    "Closure '" + param.name_ +
                    "' is not supported for inlined invoke. If you are "
                    "invoking a "
                    "function from AD (`grad`, `jacrev`, etc), please pass "
                    "`tape_in_closure=False` to it, and pass the tapes "
                    "explicitly.");
            }
        }
        for (auto &&ret : func->returns_) {
            if (ret.isInClosure()) {
                throw InvalidProgram(
                    "Closure '" + ret.name_ +
                    "' is not supported for inlined invoke. If you are "
                    "invoking a "
                    "function from AD (`grad`, `jacrev`, etc), please pass "
                    "`tape_in_closure=False` to it, and pass the tapes "
                    "explicitly.");
            }
        }
    }

    auto kvs = _kvs;
    for (auto &&[param, arg] : views::zip(func->params_, args)) {
        kvs[param.name_] = arg;
    }
    if (kvs.size() != func->params_.size()) {
        throw InvalidProgram(func->name_ + " has " +
                             std::to_string(func->params_.size()) +
                             " parameters, but " + std::to_string(kvs.size()) +
                             " arguments are provided");
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
                        kvs, renameRets, conflictNames)(ast);

    return ast;
}

} // namespace freetensor
