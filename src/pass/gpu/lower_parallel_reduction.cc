#include <pass/gpu/lower_parallel_reduction.h>

namespace ir {

namespace gpu {

uint64_t LowerParallelReduction::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

Stmt LowerParallelReduction::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    auto ret = Mutator::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

Stmt LowerParallelReduction::visit(const For &op) {
    for (auto &&[redOp, expr] : op->property_.reductions_) {
        auto h = getHash(expr);
        if (expr2for_.count(h)) {
            ERROR(
                "Parallel reduction over multiple scopes is not supported yet");
        }
        expr2for_[h] = op;
    }

    auto ret = Mutator::visit(op);

    for (auto &&[redOp, expr] : op->property_.reductions_) {
        auto h = getHash(expr);
        expr2for_.erase(h);
        auto workspace = "__reduce" + std::to_string(h);
        ASSERT(expr->nodeType() == ASTNodeType::Load);
        auto dtype = buffers_.at(expr.as<LoadNode>()->var_)->tensor().dtype();
        ret = makeVarDef("", workspace,
                         Buffer(Tensor({op->len_}, dtype), AccessType::Cache,
                                MemType::GPUShared),
                         nullptr, ret, false);
    }

    return ret;
}

Stmt LowerParallelReduction::visit(const If &op) {
    if (!expr2for_.empty()) {
        ERROR("Parallel reduction inside an `if` is not supported yet");
    }
    return Mutator::visit(op);
}

Stmt LowerParallelReduction::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    auto h = getHash(makeLoad(op->var_, op->indices_));
    if (expr2for_.count(h)) {
        auto &&loop = expr2for_.at(h);
        if (loop->len_->nodeType() != ASTNodeType::IntConst) {
            ERROR("Parallel reduction on a dynamic-lengthed loop is not "
                  "supported yet");
        }
        auto len = loop->len_.as<IntConstNode>()->val_;
        auto nth = makeSub(makeVar(loop->iter_), loop->begin_);
        auto workspace = "__reduce" + std::to_string(h);
        std::vector<Stmt> stmts;
        // workspace[nth] = expr
        stmts.emplace_back(makeStore("", workspace, {nth}, op->expr_));
        for (int k = 1; k < len; k <<= 1) {
            // if (nth % k == 0) workspace[nth] += workspace[nth + k]
            stmts.emplace_back(makeIf(
                "", makeEQ(makeMod(nth, makeIntConst(k << 1)), makeIntConst(0)),
                makeReduceTo(
                    "", workspace, {nth}, op->op_,
                    makeLoad(workspace, {makeAdd(nth, makeIntConst(k))}),
                    false)));
        }
        stmts.emplace_back(makeReduceTo("", op->var_, op->indices_, op->op_,
                                        makeLoad(workspace, {makeIntConst(0)}),
                                        false));
        return makeStmtSeq("", std::move(stmts));
    }

    return op;
}

} // namespace gpu

} // namespace ir
