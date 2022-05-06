#include <schedule/var_merge.h>

namespace freetensor {

Stmt VarMerge::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        found_ = true;

        if (dim_ + 1 >= (int)_op->buffer_->tensor()->shape().size()) {
            throw InvalidSchedule(
                "There is no dimension " + std::to_string(dim_) + " ~ " +
                std::to_string(dim_ + 1) + " in variable " + _op->name_);
        }
        factor_ = _op->buffer_->tensor()->shape()[dim_ + 1];
        var_ = _op->name_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        var_.clear();

        if (op->buffer_->atype() != AccessType::Cache &&
            !op->ioTensor_.isValid()) {
            op->ioTensor_ = op->buffer_->tensor();
        }

        auto &shape = op->buffer_->tensor()->shape();
        shape[dim_] = makeMul(shape[dim_], shape[dim_ + 1]);
        shape.erase(shape.begin() + dim_ + 1);
        return op;
    } else {
        return Mutator::visit(_op);
    }
}

Stmt VarMerge::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return mergeMemAcc(op);
}

Stmt VarMerge::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return mergeMemAcc(op);
}

Expr VarMerge::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return mergeMemAcc(op);
}

Stmt varMerge(const Stmt &_ast, const ID &def, int dim) {
    VarMerge mutator(def, dim);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule(toString(def) + " not found");
    }
    return ast;
}

} // namespace freetensor
