#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/deps.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>

namespace ir {

namespace gpu {

void FindAllThreads::visit(const For &op) {
    if (op->parallel_ == "threadIdx.x") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thx_ = op->len_.as<IntConstNode>()->val_;
    } else if (op->parallel_ == "threadIdx.y") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thy_ = op->len_.as<IntConstNode>()->val_;
    } else if (op->parallel_ == "threadIdx.z") {
        ASSERT(op->len_->nodeType() == ASTNodeType::IntConst);
        thz_ = op->len_.as<IntConstNode>()->val_;
    }
    Visitor::visit(op);
    if (op->parallel_ == "threadIdx.x") {
        results_.emplace_back(ThreadInfo{op, thx_ <= warpSize_});
    } else if (op->parallel_ == "threadIdx.y") {
        results_.emplace_back(ThreadInfo{op, thx_ * thy_ <= warpSize_});
    } else if (op->parallel_ == "threadIdx.z") {
        results_.emplace_back(ThreadInfo{op, thx_ * thy_ <= warpSize_});
    }
}

Stmt MakeSync::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = Mutator::visitStmt(op, visitNode);
    bool needSyncThreads = false, needSyncWarp = false;
    for (CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.later_.id() == op->id()) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
        }
    }
    if (needSyncThreads) {
        ret = makeStmtSeq("", {makeEval("", makeIntrinsic("__syncthreads()", {},
                                                          DataType::Void)),
                               ret});
    } else if (needSyncWarp) {
        ret = makeStmtSeq("", {makeEval("", makeIntrinsic("__syncwarp()", {},
                                                          DataType::Void)),
                               ret});
    }
    for (CrossThreadDep &dep : deps_) {
        if (dep.visiting_) {
            if (needSyncThreads) {
                dep.synced_ = true;
            }
            if (needSyncWarp && dep.inWarp_) {
                dep.synced_ = true;
            }
        }
    }
    for (CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && !dep.visiting_ && dep.earlier_.id() == op->id()) {
            dep.visiting_ = true;
        }
    }
    return ret;
}

Stmt MakeSync::visit(const For &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    bool needSyncThreads = false, needSyncWarp = false;
    for (CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.lcaLoop_.id() == op->id()) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
        }
    }
    if (needSyncThreads) {
        op->body_ = makeStmtSeq(
            "", {op->body_, makeEval("", makeIntrinsic("__syncthreads()", {},
                                                       DataType::Void))});
    } else if (needSyncWarp) {
        op->body_ = makeStmtSeq(
            "", {op->body_, makeEval("", makeIntrinsic("__syncwarp()", {},
                                                       DataType::Void))});
    }
    for (CrossThreadDep &dep : deps_) {
        if (dep.visiting_) {
            if (needSyncThreads) {
                dep.synced_ = true;
            }
            if (needSyncWarp && dep.inWarp_) {
                dep.synced_ = true;
            }
        }
    }
    return op;
}

Stmt makeSync(const Stmt &_op) {
    auto op = _op;
    FindAllThreads finder;
    finder(op);
    auto &&threads = finder.results();

    std::vector<std::vector<std::pair<std::string, DepDirection>>> query;
    query.reserve(threads.size());
    for (auto &&thr : threads) {
        query.push_back({{thr.loop_->id(), DepDirection::Different}});
    }
    std::vector<CrossThreadDep> deps;
    auto filter = [](const AccessPoint &later, const AccessPoint &earlier) {
        return later.buffer_->mtype() == MemType::GPUGlobal ||
               later.buffer_->mtype() == MemType::GPUShared;
    };
    auto found = [&](const Dependency &d) {
        auto i = &d.cond_ - &query[0];
        auto lcaLoop = lca(d.later_.cursor_, d.earlier_.cursor_);
        while (lcaLoop.hasOuter() && lcaLoop.nodeType() != ASTNodeType::For) {
            lcaLoop = lcaLoop.outer();
        }
        ASSERT(lcaLoop.nodeType() == ASTNodeType::For);
        deps.emplace_back(CrossThreadDep{d.later_.cursor_, d.earlier_.cursor_,
                                         lcaLoop, threads.at(i).inWarp_, false,
                                         false});
    };
    findDeps(op, query, found, FindDepsMode::Dep, DEP_ALL, filter, false,
             false);

    MakeSync mutator(std::move(deps));
    op = mutator(op);

    return flattenStmtSeq(op);
}

} // namespace gpu

} // namespace ir

