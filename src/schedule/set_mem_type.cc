#include <schedule/set_mem_type.h>

namespace ir {

Stmt SetMemType::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (op->id() == def_) {
        op->buffer_->setMtype(mtype_);
        found_ = true;
    }
    return op;
}

} // namespace ir
