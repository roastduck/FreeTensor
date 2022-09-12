#include <algorithm>

#include <analyze/all_uses.h>
#include <pass/make_const_shape.h>
#include <pass/pb_simplify.h>

namespace freetensor {

bool MakeConstShape::isConstOrByValue(const Expr &x) const {
    return isConstOrByValue(allNames(x));
}

bool MakeConstShape::isConstOrByValue(
    const std::unordered_set<std::string> &names) const {
    for (auto &&item : names) {
        if (hasLoop(item)) {
            // TODO: For CUDA, we should allow using iterators defined outside
            // of a kernel
            return false;
        }
        if (hasDef(item)) {
            if (buffer(item)->mtype() != MemType::ByValue) {
                return false;
            }
        }
    }
    return true;
}

Stmt MakeConstShape::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    if (std::find(mtypes_.begin(), mtypes_.end(), op->buffer_->mtype()) ==
        mtypes_.end()) {
        return op;
    }

    size_t ndim = op->buffer_->tensor()->shape().size();
    for (size_t i = 0; i < ndim; i++) {
        auto &dim = op->buffer_->tensor()->shape()[i];
        const Expr &oldDim = _op->buffer_->tensor()->shape()[i];
        if (isConstOrByValue(dim)) {
            continue;
        }
        Expr result;
        for (auto b : unique_.getUpper(oldDim)) {
            if (isConstOrByValue(b.lin().allNames())) {
                result =
                    result.isValid() ? makeMin(result, b.expr()) : b.expr();
            }
        }
        if (!result.isValid()) {
            throw InvalidProgram("Unable to relax dimension " +
                                 std::to_string(i) + ": " + toString(dim) +
                                 " of " + toString(op->id()) + "(" +
                                 toString(op->metadata()) + "): " + op->name_ +
                                 " to a constant");
        }
        dim = std::move(result);
        op->pinned_ = true;
    }
    return op;
}

Stmt makeConstShape(const Stmt &_op, const std::vector<MemType> &mtypes) {
    return MakeConstShape(mtypes)(_op);
}

} // namespace freetensor
