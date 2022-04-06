#include <pass/make_1d_var.h>
#include <pass/simplify.h>

namespace ir {

Stmt Make1DVar::visit(const VarDef &_op) {
    if (_op->buffer_->tensor().shape().size() <= 1) {
        return BaseClass::visit(_op);
    }

    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    Expr len;
    if (op->sizeLim_.isValid()) {
        len = op->sizeLim_;
    } else {
        for (Expr dim : op->buffer_->tensor().shape()) {
            len = len.isValid() ? makeMul(len, dim) : dim;
        }
    }
    op->buffer_->tensor().setShape({len});
    return op;
}

Stmt Make1DVar::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return visitMemAcc(op);
}

Stmt Make1DVar::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return visitMemAcc(op);
}

Expr Make1DVar::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return visitMemAcc(op);
}

Stmt make1dVar(const Stmt &_op) {
    auto op = Make1DVar()(_op);
    op = simplifyPass(op);
    return op;
}

} // namespace ir
