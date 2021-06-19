#include <algorithm>
#include <climits>
#include <sstream>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/gpu/make_sync.h>
#include <pass/move_out_first_or_last_iter.h>

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

Stmt CopyPart::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    if (!ended_ && end_.isValid() && op->id() == end_->id()) {
        ended_ = true;
    }
    if (ended_) {
        return makeStmtSeq("", {});
    }
    auto ret = Mutator::visitStmt(op, visitNode);
    if (!begun_) {
        ret = makeStmtSeq("", {});
    }
    if (!begun_ && begin_.isValid() && op->id() == begin_->id()) {
        begun_ = true;
    }
    return ret;
}

Stmt MakeSync::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = MutatorWithCursor::visitStmt(op, visitNode);
    Cursor target;
    bool needSyncThreads = false, needSyncWarp = false;
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.later_.node() == op) {
            (dep.inWarp_ ? needSyncWarp : needSyncThreads) = true;
            target =
                target.isValid() ? lca(target, dep.lcaStmt_) : dep.lcaStmt_;
        }
    }
    if (needSyncThreads || needSyncWarp) {
        Stmt sync;
        if (needSyncThreads) {
            sync = makeEval(
                "", makeIntrinsic("__syncthreads()", {}, DataType::Void));
        } else {
            sync =
                makeEval("", makeIntrinsic("__syncwarp()", {}, DataType::Void));
        }

        bool metVarDef = false, metFor = false;
        for (auto ctx = cursor(); ctx.id() != target.id(); ctx = ctx.outer()) {
            switch (ctx.nodeType()) {
            case ASTNodeType::For:
                metFor = true;
                sync = makeIf("",
                              makeEQ(makeVar(ctx.node().as<ForNode>()->iter_),
                                     ctx.node().as<ForNode>()->begin_),
                              sync);
                break;

            case ASTNodeType::If:
                if (metVarDef) {
                    ERROR("Cannot split a VarDef"); // TODO
                }
                if (metFor) {
                    ERROR("Cannot split a For"); // TODO
                }
                ASSERT(!ctx.node().as<IfNode>()->elseCase_.isValid()); // TODO
                branchSplitters_[ctx.id()].emplace_back(sync);
                break;

            case ASTNodeType::VarDef:
                metVarDef = true;
                break;

            case ASTNodeType::StmtSeq:
                break;

            default:
                ERROR("Unrecognized stmt " + toString(ctx.nodeType()));
            }
        }
        ret = makeStmtSeq("", {sync, ret});

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
    }
    for (CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && !dep.visiting_ && dep.earlier_.node() == op) {
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
    for (const CrossThreadDep &dep : deps_) {
        if (!dep.synced_ && dep.visiting_ && dep.lcaLoop_.node() == _op) {
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
    if (needSyncThreads || needSyncWarp) {
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
    }
    return op;
}

Stmt MakeSync::visit(const If &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();

    if (branchSplitters_.count(op->id())) {
        if (!checkNotModified(root_, op->cond_, CheckNotModifiedSide::Before,
                              op->id(), CheckNotModifiedSide::After,
                              op->id())) {
            throw InvalidProgram("Unable to insert a synchronizing statment "
                                 "inside an If node because the condition " +
                                 toString(op->cond_) + " is being modified");
        }

        auto &&splitters = branchSplitters_.at(op->id());
        std::vector<Stmt> stmts;
        stmts.reserve(splitters.size() * 2 + 1);
        for (size_t i = 0, iEnd = splitters.size() + 1; i < iEnd; i++) {
            Stmt begin = i == 0 ? nullptr : splitters[i - 1];
            Stmt end = i == iEnd - 1 ? nullptr : splitters[i];
            auto part = CopyPart(begin, end)(op->thenCase_);
            stmts.emplace_back(makeIf("", op->cond_, part));
            if (i < iEnd - 1) {
                stmts.emplace_back(splitters[i]);
            }
        }
        return makeStmtSeq("", std::move(stmts));
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
        if (later.op_ == earlier.op_) {
            return false;
        }
        return later.buffer_->mtype() == MemType::GPUGlobal ||
               later.buffer_->mtype() == MemType::GPUShared;
    };
    auto found = [&](const Dependency &d) {
        auto i = &d.cond_ - &query[0];
        auto lcaStmt = lca(d.later_.cursor_, d.earlier_.cursor_);
        auto lcaLoop = lcaStmt;
        while (lcaLoop.hasOuter() && lcaLoop.nodeType() != ASTNodeType::For) {
            lcaLoop = lcaLoop.outer();
        }
        ASSERT(lcaLoop.nodeType() == ASTNodeType::For);
        deps.emplace_back(CrossThreadDep{d.later_.cursor_, d.earlier_.cursor_,
                                         lcaStmt, lcaLoop,
                                         threads.at(i).inWarp_, false, false});
    };
    findDeps(op, query, found, FindDepsMode::Dep, DEP_ALL, filter, false,
             false);

    MakeSync mutator(op, std::move(deps));
    op = mutator(op);

    op = moveOutFirstOrLastIter(op);

    return flattenStmtSeq(op);
}

} // namespace gpu

} // namespace ir

