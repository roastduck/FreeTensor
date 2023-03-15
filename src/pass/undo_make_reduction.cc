#include <pass/undo_make_reduction.h>

namespace freetensor {

Stmt undoMakeReduction(const ReduceTo &op, DataType dtype) {
    switch (op->op_) {
    case ReduceOp::Add:
        return makeStore(
            op->var_, op->indices_,
            makeAdd(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::Sub:
        return makeStore(
            op->var_, op->indices_,
            makeSub(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::Mul:
        return makeStore(
            op->var_, op->indices_,
            makeMul(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::RealDiv:
        return makeStore(
            op->var_, op->indices_,
            makeRealDiv(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::Min:
        return makeStore(
            op->var_, op->indices_,
            makeMin(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::Max:
        return makeStore(
            op->var_, op->indices_,
            makeMax(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::LAnd:
        return makeStore(
            op->var_, op->indices_,
            makeLAnd(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    case ReduceOp::LOr:
        return makeStore(
            op->var_, op->indices_,
            makeLOr(makeLoad(op->var_, op->indices_, dtype), op->expr_),
            op->metadata(), op->id());
    default:
        ASSERT(false);
    }
}

Stmt UndoMakeReduction::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    return undoMakeReduction(op, buffer(op->var_)->tensor()->dtype());
}

} // namespace freetensor
