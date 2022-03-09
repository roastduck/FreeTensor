#include <schedule/check_var_cross_parallel.h>
#include <schedule/set_mem_type.h>

namespace ir {

Stmt SetMemType::visit(const For &op) {
    if (op->property_.parallel_.empty()) {
        return Mutator::visit(op);
    } else {
        inScope_[op->property_.parallel_]++;
        auto ret = Mutator::visit(op);
        inScope_[op->property_.parallel_]--;
        return ret;
    }
}

Stmt SetMemType::visit(const VarDef &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    if (op->id() == def_) {
        if ((mtype_ == MemType::GPUWarp || mtype_ == MemType::GPUShared ||
             mtype_ == MemType::GPULocal) &&
            inScope_["threadIdx.x"] == 0 && inScope_["threadIdx.y"] == 0 &&
            inScope_["threadIdx.z"] == 0 && inScope_["blockIdx.x"] == 0 &&
            inScope_["blockIdx.y"] == 0 && inScope_["blockIdx.z"] == 0) {
            // This restriction is brought by CodeGen, which can not be checked
            // via checkVarCrossParallel. If we improve CodeGen in the future,
            // we can lift this restriction
            throw InvalidSchedule("Unable to allocate a " + toString(mtype_) +
                                  " outside a kernel");
        }
        op->buffer_->setMtype(mtype_);
        found_ = true;
    }
    return op;
}

Stmt setMemType(const Stmt &_ast, const ID &def, MemType mtype) {
    SetMemType mutator(def, mtype);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule(toString(def) + " not found");
    }
    checkVarCrossParallel(ast, def, mtype);
    return ast;
}

} // namespace ir
