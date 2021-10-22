#include <pass/cpu/lower_parallel_reduction.h>

namespace ir {

namespace cpu {

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
    if (_op->property_.reductions_.empty()) {
        return Mutator::visit(_op);
    }

    std::vector<int64_t> hs;
    std::vector<Load> loads;
    std::vector<DataType> dtypes;
    std::vector<std::string> workspaces;
    hs.reserve(_op->property_.reductions_.size());
    loads.reserve(_op->property_.reductions_.size());
    dtypes.reserve(_op->property_.reductions_.size());
    workspaces.reserve(_op->property_.reductions_.size());
    for (auto &&[redOp, expr] : _op->property_.reductions_) {
        auto h = getHash(expr);
        if (expr2for_.count(h)) {
            ERROR(
                "Parallel reduction over multiple scopes is not supported yet");
        }
        expr2for_.insert(h);
        ASSERT(expr->nodeType() == ASTNodeType::Load);

        hs.emplace_back(h);
        loads.emplace_back(expr.as<LoadNode>());
        dtypes.emplace_back(buffers_.at(loads.back()->var_)->tensor().dtype());
        workspaces.emplace_back("__reduce" + std::to_string(h));
    }

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        expr2for_.erase(hs[i]);
        op->property_.reductions_[i].second = makeLoad(workspaces[i], {});
    }

    std::vector<Stmt> stmts;
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        stmts.emplace_back(makeStore(
            "", workspaces[i], {},
            neutralVal(dtypes[i], op->property_.reductions_[i].first)));
    }

    stmts.emplace_back(op);

    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        stmts.emplace_back(makeReduceTo("", loads[i]->var_, loads[i]->indices_,
                                        op->property_.reductions_[i].first,
                                        makeLoad(workspaces[i], {}), false));
    }

    Stmt ret = makeStmtSeq("", std::move(stmts));
    for (size_t i = 0, n = op->property_.reductions_.size(); i < n; i++) {
        ret = makeVarDef(
            "", workspaces[i],
            Buffer(Tensor({}, dtypes[i]), AccessType::Cache, MemType::CPU),
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
        auto workspace = "__reduce" + std::to_string(h);
        return makeReduceTo(op->id(), workspace, {}, op->op_, op->expr_, false);
    }

    return op;
}

} // namespace cpu

} // namespace ir
