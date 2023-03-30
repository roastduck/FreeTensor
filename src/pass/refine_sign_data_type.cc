#include <pass/refine_sign_data_type.h>
#include <pass/undo_make_reduction.h>

namespace freetensor {

Stmt ClearDataType::visit(const VarDef &_op) {
    if (!_op->viewOf_.has_value()) {
        oldTypes_[_op->id()] = _op->buffer_->tensor()->dtype();
    }
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    auto fin = op;
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    if (!isInputting(fin->buffer_->atype())) {
        op->buffer_->tensor()->setDType(
            {op->buffer_->tensor()->dtype().base(), SignDataType::Never});
    }
    return op;
}

Expr ClearDataType::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    auto fin = def(op->var_);
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    if (!isInputting(fin->buffer_->atype())) {
        op->loadType_ = {op->loadType_.base(), SignDataType::Never};
    }
    return op;
}

Stmt UpdateDType::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    auto fin = op;
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    auto dtype = fin->buffer_->tensor()->dtype();
    // Don't treat Never as dead code here. The iterating algorithm has not
    // finished yet.
    op->buffer_->tensor()->setDType(dtype);
    return op;
}

Expr UpdateDType::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    auto fin = def(op->var_);
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    op->loadType_ = fin->buffer_->tensor()->dtype();
    return op;
}

Stmt RefineSignDataType::visit(const VarDef &_op) {
    // In case of views, we acount all access to the final viewee. The viewer is
    // updated after the viewee
    auto oldType = _op->buffer_->tensor()->dtype();
    if (!_op->viewOf_.has_value()) {
        newTypes_[_op->id()] = oldType;
    }

    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (!_op->viewOf_.has_value()) {
        auto newType =
            downCast(newTypes_.at(op->id()), userTypes_.at(op->id()));
        if (newType.sign() != oldType.sign()) {
            converged_ = false;
            op->buffer_->tensor()->setDType({oldType.base(), newType.sign()});
        }
    }
    return op;
}

Stmt RefineSignDataType::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    auto fin = def(op->var_);
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    newTypes_[fin->id()] = upCast(newTypes_.at(fin->id()), op->expr_->dtype());
    return op;
}

Stmt RefineSignDataType::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    auto fin = def(op->var_);
    while (fin->viewOf_.has_value()) {
        fin = def(*fin->viewOf_);
    }
    (*this)(undoMakeReduction(op, newTypes_.at(fin->id())));
    return op;
}

Stmt refineSignDataType(const Stmt &_op) {
    ClearDataType clear;
    auto op = clear(_op);
    for (int i = 0;; i++) {
        RefineSignDataType mutator{clear.oldTypes()};
        op = mutator(op);
        op = UpdateDType{}(op);
        if (mutator.converged() || i > 100) {
            if (i > 100) {
                WARNING("RefineSignDataType iterates over 100 rounds. Maybe "
                        "there is a bug");
            }
            break;
        }
    }
    return op;
}

} // namespace freetensor
