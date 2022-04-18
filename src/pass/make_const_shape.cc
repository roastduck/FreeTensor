#include <algorithm>
#include <limits>

#include <pass/make_const_shape.h>
#include <pass/pb_simplify.h>

namespace ir {

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
        if (dim->nodeType() == ASTNodeType::IntConst) {
            continue;
        }
        int64_t result = std::numeric_limits<int64_t>::max();
        for (auto b : unique_.getUpper(oldDim)) {
            if (b.lin().isConst()) {
                auto bias = b.lin().bias_;
                result = std::min(result, floorDiv(bias.p_, bias.q_));
            }
        }
        if (result == std::numeric_limits<int64_t>::max()) {
            throw InvalidProgram("Unable to relax dimension " +
                                 std::to_string(i) + ": " + toString(dim) +
                                 " of " + op->id().strId() + ": " + op->name_ +
                                 " to a constant");
        }
        dim = makeIntConst(result);
        op->pinned_ = true;
    }
    return op;
}

Stmt makeConstShape(const Stmt &_op, const std::vector<MemType> &mtypes) {
    return MakeConstShape(mtypes)(_op);
}

} // namespace ir
