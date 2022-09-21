#include <schedule.h>
#include <schedule/var_split.h>

namespace freetensor {

Stmt VarSplit::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        found_ = true;

        if (dim_ >= (int)_op->buffer_->tensor()->shape().size()) {
            throw InvalidSchedule("There is no dimension " +
                                  std::to_string(dim_) + " in variable " +
                                  _op->name_);
        }
        if (factor_ != -1) {
            dynFactor_ = makeIntConst(factor_);
        } else {
            ASSERT(nparts_ != -1);
            dynFactor_ = makeCeilDiv(_op->buffer_->tensor()->shape()[dim_],
                                     makeIntConst(nparts_));
        }

        var_ = _op->name_;
        newVar_ =
            _op->buffer_->atype() == AccessType::Cache ? var_ : var_ + ".view";
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        var_.clear();
        newVar_.clear();

        auto &shape = op->buffer_->tensor()->shape();
        if (factor_ != -1) {
            shape[dim_] = makeCeilDiv(shape[dim_], dynFactor_);
            shape.insert(shape.begin() + dim_ + 1, dynFactor_);
        } else {
            ASSERT(nparts_ != -1);
            shape[dim_] = dynFactor_;
            shape.insert(shape.begin() + dim_, makeIntConst(nparts_));
        }

        if (op->buffer_->atype() != AccessType::Cache) {
            if (fixedSize_) {
                op->name_ += ".view";
                op->viewOf_ = _op->name_;
                op->buffer_->setAtype(AccessType::Cache);
                return makeVarDef(_op->name_, _op->buffer_, std::nullopt, op,
                                  false);
            } else {
                throw InvalidSchedule(
                    "Using RelaxedSize mode in an I/O variable is not allowed");
            }
        } else {
            return op;
        }
    } else {
        return Mutator::visit(_op);
    }
}

Stmt VarSplit::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();
    return splitMemAcc(op);
}

Stmt VarSplit::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();
    return splitMemAcc(op);
}

Expr VarSplit::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    return splitMemAcc(op);
}

Stmt varSplit(const Stmt &_ast, const ID &def, int dim, VarSplitMode mode,
              int factor, int nparts) {
    VarSplit mutator(def, dim, mode == VarSplitMode::FixedSize, factor, nparts);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule(toString(def) + " not found");
    }
    return ast;
}

void Schedule::varSplit(const ID &def, int dim, VarSplitMode mode, int factor,
                        int nparts) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(VarSplit, freetensor::varSplit, def,
                                           dim, mode, factor, nparts));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
