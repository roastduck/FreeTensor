#include <analyze/check_not_modified.h>
#include <pass/hoist_return_vars.h>

namespace freetensor {

Stmt HoistReturnVars::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (outMostLoop_.isValid() &&
        std::find_if(func_->returns_.begin(), func_->returns_.end(),
                     [&](const FuncRet &ret) {
                         return ret.name_ == op->name_;
                     }) != func_->returns_.end()) {
        for (auto &&dim : op->buffer_->tensor()->shape()) {
            if (!checkNotModified(func_->body_, dim,
                                  CheckNotModifiedSide::Before, op->id(),
                                  CheckNotModifiedSide::Before, outMostLoop_)) {
                throw InvalidProgram(
                    "A `Func`'s returning values are allocated during run "
                    "time, and the allocation cannot be parallelized. "
                    "Furthermore, "
                    "it is unable to hoist " +
                    op->name_ + " out of " + toString(outMostLoop_) +
                    " or the dimension size " + toString(dim) +
                    " will be modified");
            }
        }
        toHoist_.emplace_back(op);
        return op->body_;
    }
    return op;
}

Stmt HoistReturnVars::visit(const For &op) {
    if (!outMostLoop_.isValid()) {
        outMostLoop_ = op->id();
        auto ret = Mutator::visit(op);
        outMostLoop_ = ID();

        for (auto def : toHoist_) {
            ret = makeVarDef(def->name_, def->buffer_,
                             def->ioTensor_, std::move(ret), def->pinned_,
                             def->metadata(), def->id());
        }
        return ret;
    } else {
        return Mutator::visit(op);
    }
}

Func hoistReturnVars(const Func &func) {
    return makeFunc(func->name_, func->params_, func->returns_,
                    HoistReturnVars(func)(func->body_));
}

} // namespace freetensor
