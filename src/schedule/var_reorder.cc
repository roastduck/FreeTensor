#include <analyze/all_uses.h>
#include <schedule.h>
#include <schedule/var_reorder.h>

namespace freetensor {

Stmt VarReorder::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        if (_op->buffer_->atype() != AccessType::Cache) {
            throw InvalidSchedule("Reorder on an I/O variable is not allowed");
        }

        found_ = true;

        var_ = _op->name_;
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        var_.clear();

        std::vector<Expr> shape;
        shape.reserve(order_.size());
        ASSERT(order_.size() == op->buffer_->tensor()->shape().size());
        for (size_t i = 0, n = order_.size(); i < n; i++) {
            shape.emplace_back(op->buffer_->tensor()->shape()[order_[i]]);
        }
        op->buffer_->tensor()->setShape(shape);
        return op;
    } else {
        auto source = _op;
        while (source->viewOf_.has_value()) {
            source = def(*source->viewOf_);
            if (source->id() == def_) {
                throw InvalidSchedule(
                    "Cannot var_reorder a VarDef node that has views");
            }
        }
        return BaseClass::visit(_op);
    }
}

Stmt VarReorder::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return reorderMemAcc(op);
}

Stmt VarReorder::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return reorderMemAcc(op);
}

Expr VarReorder::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return reorderMemAcc(op);
}

Stmt VarReorder::visit(const MatMul &op) {
    if (!var_.empty() && (allReads(op->equivalent_).count(var_) ||
                          allWrites(op->equivalent_).count(var_))) {
        throw InvalidSchedule("Please call var_reorder before as_matmul");
    }
    return BaseClass::visit(op);
}

Stmt varReorder(const Stmt &_ast, const ID &def,
                const std::vector<int> &order) {
    VarReorder mutator(def, order);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule(toString(def) + " not found");
    }
    return ast;
}

void Schedule::varReorder(const ID &def, const std::vector<int> &order) {
    beginTransaction();
    auto log = appendLog(
        MAKE_SCHEDULE_LOG(VarReorder, freetensor::varReorder, def, order));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
