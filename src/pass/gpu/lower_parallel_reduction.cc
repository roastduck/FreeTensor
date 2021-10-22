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

Stmt LowerParallelReduction::visit(const For &_op) {
    for (auto &&[redOp, expr] : _op->property_.reductions_) {
        auto h = getHash(expr);
        if (expr2for_.count(h)) {
            ERROR(
                "Parallel reduction over multiple scopes is not supported yet");
        }
        expr2for_[h] = _op;
    }

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    for (auto &&[redOp, expr] : op->property_.reductions_) {
        if (op->len_->nodeType() != ASTNodeType::IntConst) {
            ERROR("Parallel reduction on a dynamic-lengthed loop is not "
                  "supported yet");
        }
        auto len = op->len_.as<IntConstNode>()->val_;
        auto nth = makeSub(makeVar(op->iter_), op->begin_);

        ASSERT(expr->nodeType() == ASTNodeType::Load);
        auto load = expr.as<LoadNode>();
        auto dtype = buffers_.at(load->var_)->tensor().dtype();
        auto h = getHash(expr);
        expr2for_.erase(h);
        auto workspace = "__reduce" + std::to_string(h);

        // Note that we are before normalize_threads, so there will be no
        // cross-thread dependencies except the reduction we are working on.
        // Therefore, we don't have to immediately reduce the values at the
        // ReduceTo node, and we can deal with the reduction at the end of the
        // parallel scope.
        std::vector<Stmt> stmts;
        // workspace[nth] = 0
        stmts.emplace_back(
            makeStore("", workspace, {nth}, neutralVal(dtype, redOp)));
        // body
        stmts.emplace_back(op->body_);
        // for (int k = 1; k < len; k <<= 1)
        //   if (nth % k == 0 && nth + k < len)
        //     workspace[nth] += workspace[nth + k]
        // where k = 2^p
        //   => 2^p < len
        //   => p < log_2 len
        //   => p < floor(log_2(len - 1)) + 1
        auto count = (63 - __builtin_clzll((unsigned long long)(len - 1))) + 1;
        auto k =
            makeIntrinsic("1 << (%)", {makeVar("__reduce_p")}, DataType::Int32);
        auto reduceStmt =
            makeIf("",
                   makeLAnd(makeEQ(makeMod(nth, makeMul(k, makeIntConst(2))),
                                   makeIntConst(0)),
                            makeLT(makeAdd(nth, k), op->len_)),
                   makeReduceTo("", workspace, {nth}, redOp,
                                makeLoad(workspace, {makeAdd(nth, k)}), false));
        stmts.emplace_back(makeFor("", "__reduce_p", makeIntConst(0),
                                   makeIntConst(count), makeIntConst(count),
                                   false, ForProperty().withUnroll(),
                                   reduceStmt));
        stmts.emplace_back(makeReduceTo("", load->var_, load->indices_, redOp,
                                        makeLoad(workspace, {makeIntConst(0)}),
                                        false));
        op->body_ = makeStmtSeq("", std::move(stmts));
    }

    Stmt ret = op;
    for (auto &&[redOp, expr] : op->property_.reductions_) {
        ASSERT(expr->nodeType() == ASTNodeType::Load);
        auto load = expr.as<LoadNode>();
        auto dtype = buffers_.at(load->var_)->tensor().dtype();
        auto h = getHash(expr);
        expr2for_.erase(h);
        auto workspace = "__reduce" + std::to_string(h);

        ret = makeVarDef("", workspace,
                         Buffer(Tensor({op->len_}, dtype), AccessType::Cache,
                                MemType::GPUShared),
                         nullptr, ret, false);
    }

    return ret;
}

Stmt LowerParallelReduction::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    auto h = getHash(makeLoad(op->var_, op->indices_));
    if (expr2for_.count(h)) {
        auto &&loop = expr2for_.at(h);
        auto nth = makeSub(makeVar(loop->iter_), loop->begin_);
        auto workspace = "__reduce" + std::to_string(h);
        return makeReduceTo(op->id(), workspace, {nth}, op->op_, op->expr_,
                            false);
    }

    return op;
}

} // namespace gpu

} // namespace ir
