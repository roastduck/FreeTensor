#include <pass/gpu/normalize_var_in_kernel.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

namespace gpu {

Stmt NormalizeVarInKernel::visit(const VarDef &_op) {
    if (inKernel_) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();

        for (auto &dim : op->buffer_->tensor()->shape()) {
            Expr newDim =
                unique_.getBound(dim)->restrictScope(legalNames_)->upperExpr();
            if (!newDim.isValid()) {
                throw InvalidProgram(
                    "The shape of " + toString(op->id()) + " " + op->name_ +
                    " should be able to be determined outside a CUDA kernel");
            }
            dim = std::move(newDim);
        }

        if (op->buffer_->mtype() == MemType::GPUGlobalHeap ||
            op->buffer_->mtype() == MemType::GPUGlobal) {
            // Hoist so we are able to turn it into GPUGlobalHeap and insert
            // Alloc and Free
            varsToHoist_.emplace_back(op);
            return op->body_;
        } else {
            return op;
        }
    } else {
        legalNames_.insert(_op->name_);
        auto ret = BaseClass::visit(_op);
        legalNames_.erase(_op->name_);
        return ret;
    }
}

Stmt NormalizeVarInKernel::visit(const For &op) {
    if (!inKernel_ &&
        std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        inKernel_ = true;
        auto ret = BaseClass::visit(op);
        inKernel_ = false;

        for (auto &&def : varsToHoist_) {
            auto newRet = def;
            newRet->body_ = ret;
            ret = std::move(newRet);
        }
        varsToHoist_.clear();
        return ret;
    } else {
        legalNames_.insert(op->iter_);
        auto ret = BaseClass::visit(op);
        legalNames_.erase(op->iter_);
        return ret;
    }
}

Stmt normalizeVarInKernel(const Stmt &_op) {
    auto op = NormalizeVarInKernel{}(_op);
    return simplify(z3Simplify(op));
}

} // namespace gpu

} // namespace freetensor
