#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/normalize.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>
#include <pass/simplify.h>

namespace ir {

namespace gpu {

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

int MakeSync::getLen(const Expr &len) {
    int ret = INT_MAX;
    if (upper_.count(len)) {
        for (auto &&b : upper_.at(len)) {
            if (b.expr_->nodeType() == ASTNodeType::IntConst) {
                ret = std::min(ret, b.expr_.as<IntConstNode>()->val_);
            }
        }
    }
    return ret;
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
        thx = getLen(_op->infoLen_);
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
        thy = getLen(_op->infoLen_);
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
        thz = getLen(_op->infoLen_);
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

Stmt makeSync(const Stmt &_op) {
    Stmt op;
    std::unordered_map<Expr, std::vector<LowerBound>> lower;
    std::unordered_map<Expr, std::vector<UpperBound>> upper;
    op = normalize(_op);
    std::tie(op, lower, upper) = simplifyAndGetBounds(op);
    auto ret = MakeSync(upper)(op);
    ret = flattenStmtSeq(ret);
    return ret;
}

} // namespace gpu

} // namespace ir

