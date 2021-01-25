#include <sstream>

#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>

namespace ir {

namespace gpu {

static void checkVarLen(const std::string &parallel, const Expr &len) {
    if (len->nodeType() != ASTNodeType::IntConst) {
        std::ostringstream msg;
        msg << "Length of " << parallel << " should be constant, instead of "
            << len;
        throw Error(msg.str());
    }
}

static Stmt insertIntrin(const Stmt &op, const std::string &front,
                         const std::string &back) {
    if (front.empty() && back.empty()) {
        return op;
    }
    std::vector<Stmt> stmts = {op};
    if (!front.empty()) {
        stmts.insert(stmts.begin(), makeEval("", makeIntrinsic(front, {})));
    }
    if (!back.empty()) {
        stmts.emplace_back(makeEval("", makeIntrinsic(back, {})));
    }
    return makeStmtSeq("", std::move(stmts));
}

Stmt MakeSync::visit(const For &_op) {
    // NOTE: We need to sync both at beginning and at the end of a parallel
    // region. Consider the following example:
    //
    // for i = 0 to 32 { // in a warp
    //     shmem[...] = ...
    //     // no syncthreads
    // }
    // for i = 0 to 256 { // not in a warp
    //     __syncthreads();
    //     ... = shmem[...]
    // }

    bool oldWarpSynced = warpSynced, oldThreadsSynced = threadsSynced;

    if (_op->parallel_ == "threadIdx.x") {
        checkVarLen(_op->parallel_, _op->info_len_);
        thx = _op->info_len_.as<IntConstNode>()->val_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        if (thx <= warpSize) {
            op->body_ =
                insertIntrin(op->body_, oldWarpSynced ? "" : "__syncwarp()",
                             warpSynced ? "" : "__syncwarp()");
            warpSynced = true;
        } else {
            op->body_ = insertIntrin(op->body_,
                                     oldThreadsSynced ? "" : "__syncthreads()",
                                     threadsSynced ? "" : "__syncthreads()");
            warpSynced = threadsSynced = true;
        }
        return op;

    } else if (_op->parallel_ == "threadIdx.y") {
        checkVarLen(_op->parallel_, _op->info_len_);
        thy = _op->info_len_.as<IntConstNode>()->val_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        if (thx * thy <= warpSize) {
            op->body_ =
                insertIntrin(op->body_, oldWarpSynced ? "" : "__syncwarp()",
                             warpSynced ? "" : "__syncwarp()");
            warpSynced = true;
        } else {
            op->body_ = insertIntrin(op->body_,
                                     oldThreadsSynced ? "" : "__syncthreads()",
                                     threadsSynced ? "" : "__syncthreads()");
            warpSynced = threadsSynced = true;
        }
        return op;

    } else if (_op->parallel_ == "threadIdx.z") {
        checkVarLen(_op->parallel_, _op->info_len_);
        thz = _op->info_len_.as<IntConstNode>()->val_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        if (thx * thy * thz <= warpSize) {
            op->body_ =
                insertIntrin(op->body_, oldWarpSynced ? "" : "__syncwarp()",
                             warpSynced ? "" : "__syncwarp()");
            warpSynced = true;
        } else {
            op->body_ = insertIntrin(op->body_,
                                     oldThreadsSynced ? "" : "__syncthreads()",
                                     threadsSynced ? "" : "__syncthreads()");
            warpSynced = threadsSynced = true;
        }
        return op;

    } else {
        return Mutator::visit(_op);
    }
}

Expr MakeSync::visit(const Load &op) {
    threadsSynced = warpSynced = false;
    return Mutator::visit(op);
}

Stmt MakeSync::visit(const Store &op) {
    threadsSynced = warpSynced = false;
    return Mutator::visit(op);
}

Stmt MakeSync::visit(const ReduceTo &op) {
    threadsSynced = warpSynced = false;
    return Mutator::visit(op);
}

Stmt makeSync(const Stmt &op) {
    auto ret = MakeSync()(op);
    ret = flattenStmtSeq(ret);
    return ret;
}

} // namespace gpu

} // namespace ir

