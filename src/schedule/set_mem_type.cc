#include <schedule.h>
#include <schedule/check_var_cross_parallel.h>
#include <schedule/set_mem_type.h>

namespace freetensor {

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

Stmt setMemType(const Stmt &_ast, const ID &def, MemType mtype) {
    SetMemType mutator(def, mtype);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule(toString(def) + " not found");
    }
    checkVarCrossParallel(ast, def, mtype);
    return ast;
}

void Schedule::setMemType(const ID &def, MemType mtype) {
    beginTransaction();
    auto log = appendLog(
        MAKE_SCHEDULE_LOG(SetMemType, freetensor::setMemType, def, mtype));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
