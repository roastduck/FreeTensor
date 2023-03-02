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
    }
    return op;
}

Stmt setMemType(const Stmt &_ast, const ID &def, MemType mtype,
                bool rejectIndirectAccess) {
    if (rejectIndirectAccess) {
        ThrowIndirectAccess{def}(_ast);
    }
    SetMemType mutator(def, mtype);
    auto ast = mutator(_ast);
    checkVarCrossParallel(ast, def, mtype);
    return ast;
}

void Schedule::setMemType(const ID &def, MemType mtype,
                          bool rejectIndirectAccess) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(SetMemType, freetensor::setMemType,
                                           def, mtype, rejectIndirectAccess));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

void Schedule::setMemType(const ID &def, MemType mtype) {
    bool rejectIndirectAccess;
    switch (mtype) {
    case MemType::GPULocal:
    case MemType::GPUWarp:
        rejectIndirectAccess = true;
        break;
    default:
        rejectIndirectAccess = false;
    }
    setMemType(def, mtype, rejectIndirectAccess);
}

} // namespace freetensor
