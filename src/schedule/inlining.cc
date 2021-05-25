#include <analyze/hash.h>
#include <schedule/inlining.h>

namespace ir {

MakeInlinePlaceholder::MakeInlinePlaceholder(const std::vector<Expr> &indices) {
    indexHashes_.reserve(indices.size());
    for (auto &&index : indices) {
        indexHashes_.emplace_back(getHash(index));
    }
}

Expr MakeInlinePlaceholder::visitExpr(
    const Expr &op, const std::function<Expr(const Expr &)> &visitNode) {
    auto h = getHash(op);
    for (size_t i = 0, iEnd = indexHashes_.size(); i < iEnd; i++) {
        if (indexHashes_[i] == h) {
            return makeVar(".inline_placeholder." + std::to_string(i));
        }
    }
    return Mutator::visitExpr(op, visitNode);
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

} // namespace ir

