#include <itertools.hpp>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <hash.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <pass/replace_iter.h>
#include <pass/simplify.h>
#include <pass/sink_var.h>
#include <schedule/inlining.h>

namespace ir {

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
        return makeStmtSeq("", {});
    } else {
        return Mutator::visit(op);
    }
}

Stmt MakeInline::visit(const ReduceTo &op) {
    if (op->var_ == var_) {
        return makeStmtSeq("", {});
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
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.op_->nodeType() == ASTNodeType::Load &&
               earlier.def_->id() == def;
    };
    auto found = [&](const Dependency &dep) {
        if (replace.count(dep.later().as<LoadNode>())) {
            throw InvalidSchedule("Multiple writes correspond to one read");
        }
        Expr expr, placeholder;
        if (dep.earlier()->nodeType() == ASTNodeType::Store) {
            auto earlier = dep.earlier().as<StoreNode>();
            expr = earlier->expr_;
            if (!allIters(expr).empty()) {
                std::unordered_map<std::string, Expr> replaceAsPlaceholder;
                for (auto &&[i, idx] : iter::enumerate(earlier->indices_)) {
                    if (idx->nodeType() == ASTNodeType::Var) {
                        replaceAsPlaceholder[idx.as<VarNode>()->name_] =
                            makeVar(".inline_placeholder." + std::to_string(i));
                    } else if (!idx->isConst()) {
                        throw InvalidSchedule(
                            "Inlining a variable that is stored using "
                            "non-iterator "
                            "index " +
                            toString(idx) +
                            " is currenctly unsupported"); // TODO
                    }
                }
                placeholder = ReplaceIter(replaceAsPlaceholder)(expr);
            } else {
                placeholder = expr;
            }
        } else {
            throw InvalidSchedule(
                "Unsupported: ReduceTo nodes cannot be inlined");
        }

        auto common = lca(dep.later_.cursor_, dep.earlier_.cursor_);
        auto d = dep.dep_;
        for (auto &&iter : allIters(expr)) {
            for (auto c = common; c.isValid(); c = c.outer()) {
                if (c.nodeType() == ASTNodeType::For) {
                    if (auto &&f = c.node().as<ForNode>(); f->iter_ == iter) {
                        d = dep.extraCheck(d, f->id(), DepDirection::Same);
                        if (d != dep.dep_) {
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
        std::unordered_map<std::string, Expr> replaceFromPlaceholder;
        for (auto &&[i, idx] : iter::enumerate(later->indices_)) {
            replaceFromPlaceholder[".inline_placeholder." + std::to_string(i)] =
                idx;
        }
        auto newExpr = ReplaceIter(replaceFromPlaceholder)(placeholder);

        if (!checkNotModified(ast, expr, newExpr, CheckNotModifiedSide::Before,
                              dep.earlier_.cursor_.id(),
                              CheckNotModifiedSide::Before,
                              dep.later_.cursor_.id())) {
            throw InvalidSchedule(
                "The expression will be modified after inlining from " +
                toString(dep.earlier_.stmt_) + " into " +
                toString(dep.later_.stmt_));
        }
        replace[later] = std::move(newExpr);
    };
    findDeps(ast, {{}}, found, FindDepsMode::KillLater, DEP_RAW, filter);
    ast = MakeInline(def, replace)(ast);

    ast = sinkVar(ast);
    ast = simplifyPass(ast);

    return ast;
}

} // namespace ir
