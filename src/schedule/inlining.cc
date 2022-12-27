#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <container_utils.h>
#include <hash.h>
#include <math/parse_pb_expr.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/replace_iter.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <schedule.h>
#include <schedule/inlining.h>

namespace freetensor {

Expr MakeInline::visit(const Load &op) {
    if (op->var_ == var_) {
        if (replace_.count(op)) {
            return (*this)(replace_.at(op));
        } else {
            throw InvalidSchedule("Unable to inline into " + toString(op));
        }
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInline::visit(const Store &op) {
    if (op->var_ == var_) {
        return makeStmtSeq({});
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInline::visit(const ReduceTo &op) {
    if (op->var_ == var_) {
        return makeStmtSeq({});
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInline::visit(const VarDef &_op) {
    if (_op->id() == def_) {
        if (_op->buffer_->atype() != AccessType::Cache) {
            throw InvalidSchedule("Cannot remove an I/O variable");
        }
        var_ = _op->name_;
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        var_.clear();
        return op->body_;
    } else {
        return Mutator::visit(_op);
    }
}

Stmt inlining(const Stmt &_ast, const ID &def) {
    auto ast = _ast;

    // A new Store/ReduceTo node may contain Load nodes out of their VarDef
    // scopes, so we have to expand those VarDef nodes. We first call
    // hoistVarDefOverStmtSeq to expand the VarDef nodes over all the statment
    // in a StmtSeq, and then we call ReplaceUses to update the Store/ReduceTo
    // nodes, and finally we call sinkVars to adjust the scope of the VarDef
    // nodes back to a proper size.
    ast = hoistVarOverStmtSeq(ast);

    std::unordered_map<Load, Expr> replace;
    std::mutex m;
    auto found = [&](const Dependence &dep) {
        {
            std::lock_guard l(m);
            if (replace.count(dep.later().as<LoadNode>())) {
                throw InvalidSchedule("Multiple writes correspond to one read");
            }
        }
        Expr expr, newExpr;
        if (dep.earlier()->nodeType() == ASTNodeType::Store) {
            auto earlier = dep.earlier().as<StoreNode>();
            expr = earlier->expr_;
            if (!allIters(expr).empty()) {
                try {
                    if (!dep.later2EarlierIter_
                             .isSingleValued()) { // Check before converting
                                                  // into PBFunc
                        throw ParserError("ISL map is not single-valued");
                    }
                    auto &&[args, values, cond] = parseSimplePBFunc(
                        toString(PBFunc(dep.later2EarlierIter_)));
                    ASSERT(dep.earlier_.iter_.size() <=
                           values.size()); // maybe padded
                    ASSERT(dep.later_.iter_.size() <= args.size());
                    std::unordered_map<std::string, Expr> islVarToNewIter,
                        oldIterToNewIter;
                    for (auto &&[newIter, arg] :
                         views::zip(dep.later_.iter_, args)) {
                        islVarToNewIter[arg] =
                            !newIter.negStep_
                                ? newIter.iter_
                                : makeMul(makeIntConst(-1), newIter.iter_);
                    }
                    for (auto &&[oldIter, value] :
                         views::zip(dep.earlier_.iter_, values)) {
                        if (oldIter.iter_->nodeType() == ASTNodeType::Var) {
                            oldIterToNewIter[oldIter.iter_.as<VarNode>()
                                                 ->name_] =
                                !oldIter.negStep_
                                    ? ReplaceIter(islVarToNewIter)(value)
                                    : makeMul(
                                          makeIntConst(-1),
                                          ReplaceIter(islVarToNewIter)(value));
                        }
                    }
                    newExpr = ReplaceIter(oldIterToNewIter)(expr);
                } catch (const ParserError &e) {
                    throw InvalidSchedule(
                        "Unable to resolve relation of the iterators between "
                        "the defining point and the use point: " +
                        toString(PBFunc(dep.later2EarlierIter_)));
                }
            } else {
                newExpr = expr;
            }
        } else {
            throw InvalidSchedule(
                "Unsupported: ReduceTo nodes cannot be inlined");
        }

        auto common = lcaStmt(dep.later_.stmt_, dep.earlier_.stmt_);
        auto d = dep.later2EarlierIter_;
        for (auto &&iter : allIters(expr)) {
            for (auto c = common; c.isValid(); c = c->parentStmt()) {
                if (c->nodeType() == ASTNodeType::For) {
                    if (auto &&f = c.as<ForNode>(); f->iter_ == iter) {
                        d = dep.extraCheck(d, f->id(), DepDirection::Same);
                        if (d != dep.later2EarlierIter_) {
                            throw InvalidSchedule(
                                "Unsupported: The loop iterator will be "
                                "changed after inlining from " +
                                toString(dep.earlier_.stmt_) + " into " +
                                toString(dep.later_.stmt_));
                        }
                        break;
                    }
                }
            }
        }

        auto later = dep.later().as<LoadNode>();
        if (!checkNotModified(ast, expr, newExpr, CheckNotModifiedSide::Before,
                              dep.earlier_.stmt_->id(),
                              CheckNotModifiedSide::Before,
                              dep.later_.stmt_->id())) {
            throw InvalidSchedule(
                "The expression will be modified after inlining from " +
                toString(dep.earlier_.stmt_) + " into " +
                toString(dep.later_.stmt_));
        }
        {
            std::lock_guard l(m);
            replace[later] = std::move(newExpr);
        }
    };
    FindDeps()
        .mode(FindDepsMode::KillLater)
        .type(DEP_RAW)
        .filterAccess([&](const auto &acc) { return acc.def_->id() == def; })
        .filterLater([&](const auto &later) {
            return later.op_->nodeType() == ASTNodeType::Load;
        })
        .noProjectOutPrivateAxis(true)(ast, unsyncFunc(found));
    ast = MakeInline(def, replace)(ast);

    ast = sinkVar(ast);
    ast = simplify(ast);

    return ast;
}

void Schedule::inlining(const ID &def) {
    beginTransaction();
    auto log = appendLog(MAKE_SCHEDULE_LOG(Inline, freetensor::inlining, def));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
