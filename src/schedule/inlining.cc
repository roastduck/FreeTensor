#include <itertools.hpp>

#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <pass/simplify.h>
#include <schedule/inlining.h>

namespace ir {

MakeInlinePlaceholder::MakeInlinePlaceholder(const std::vector<Expr> &indices) {
    indexHashes_.reserve(indices.size());
    for (auto &&index : indices) {
        indexHashes_.emplace_back(index->hash());
    }
}

Expr MakeInlinePlaceholder::visitExpr(const Expr &op) {
    auto h = op->hash();
    for (auto &&[i, indexHash] : iter::enumerate(indexHashes_)) {
        if (indexHash == h) {
            return makeVar(".inline_placeholder." + std::to_string(i));
        }
    }
    return Mutator::visitExpr(op);
}

Expr ApplyInlinePlaceholder::visit(const Var &op) {
    if (op->name_.substr(0, 20) == ".inline_placeholder.") {
        int pos = std::stoi(op->name_.substr(20));
        return indices_.at(pos);
    }
    return Mutator::visit(op);
}

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

Stmt inlining(const Stmt &_ast, const std::string &def) {
    std::unordered_map<Load, Expr> replace;
    auto filter = [&](const AccessPoint &later, const AccessPoint &earlier) {
        return later.op_->nodeType() == ASTNodeType::Load &&
               earlier.def_->id() == def;
    };
    auto found = [&](const Dependency &dep) {
        if (replace.count(dep.later().as<LoadNode>())) {
            throw InvalidSchedule("Multiple writes correspond to one read");
        }
        Expr expr;
        if (dep.earlier()->nodeType() == ASTNodeType::Store) {
            auto earlier = dep.earlier().as<StoreNode>();
            expr = MakeInlinePlaceholder(earlier->indices_)(earlier->expr_);
        } else {
            ASSERT(dep.earlier()->nodeType() == ASTNodeType::ReduceTo);
            auto earlier = dep.earlier().as<ReduceToNode>();
            expr = MakeInlinePlaceholder(earlier->indices_)(earlier->expr_);
        }
        if (!checkNotModified(_ast, expr, CheckNotModifiedSide::After,
                              dep.earlier_.cursor_.id(),
                              CheckNotModifiedSide::Before,
                              dep.later_.cursor_.id())) {
            throw InvalidSchedule(
                "The expression will be modified after inlining from " +
                toString(dep.earlier_.cursor_.node()) + " into " +
                toString(dep.later_.cursor_.node()));
        }
        auto later = dep.later().as<LoadNode>();
        replace[later] = ApplyInlinePlaceholder(later->indices_)(expr);
    };
    findDeps(_ast, {{}}, found, FindDepsMode::KillLater, DEP_RAW, filter);
    auto ast = MakeInline(def, replace)(_ast);
    ast = simplifyPass(ast);
    return ast;
}

} // namespace ir

