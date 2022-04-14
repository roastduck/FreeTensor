#include <frontend/inlined_invoke.h>
#include <pass/undo_make_reduction.h>

namespace ir {

Stmt InlinedInvoke::visitStmt(const Stmt &op) {
    auto ret = Mutator::visitStmt(op);
    ret->setId(callSiteId_.strId() + "/" + ret->id().strId());
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
            .asStore(op->id(), op->expr_);
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

Stmt inlinedInvoke(
    const ID &callSiteId, const Func &func,
    const std::vector<Ref<FrontendVar>> &args,
    const std::unordered_map<std::string, Ref<FrontendVar>> &_kvs) {
    Stmt ast = func->body_;
    ast = undoMakeReduction(ast);

    auto kvs = _kvs;
    if (args.size() > func->params_.size()) {
        throw InvalidProgram(func->name_ + " has only " +
                             std::to_string(func->params_.size()) +
                             " parameters, but " + std::to_string(args.size()) +
                             " arguments are provided");
    }
    for (size_t i = 0, n = args.size(); i < n; i++) {
        kvs[func->params_[i]] = args[i];
    }
    ast = InlinedInvoke(callSiteId, kvs)(ast);

    return ast;
}

} // namespace ir
