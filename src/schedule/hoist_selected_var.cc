#include <analyze/all_uses.h>
#include <analyze/find_stmt.h>
#include <container_utils.h>
#include <pass/rename_var.h>
#include <schedule/hoist_selected_var.h>

namespace freetensor {

static std::string getNewName(const std::string &oldName,
                              const std::unordered_set<std::string> &used) {
    for (int i = 1;; i++) {
        if (auto name = oldName + "." + std::to_string(i); !used.count(name)) {
            return name;
        }
    }
}

Stmt HoistSelectedVar::visit(const For &op) {
    if (toHoist_.count(op->body_->id())) {
        ASSERT(op->body_->nodeType() == ASTNodeType::VarDef);
        auto def = op->body_.as<VarDefNode>();
        for (auto &&dim : def->buffer_->tensor()->shape()) {
            if (allIters(dim).count(op->iter_)) {
                throw InvalidSchedule("Impossible to hoist VarDef " +
                                      toString(def->id()) + " over loop " +
                                      toString(op->id()) +
                                      " or its shape would be invalid");
            }
        }
        // recurse after making node, to maintain symbol table
        auto ret = makeVarDef(def->name_, def->buffer_, def->viewOf_,
                              makeFor(op->iter_, op->begin_, op->end_,
                                      op->step_, op->len_, op->property_,
                                      def->body_, op->metadata(), op->id()),
                              def->pinned_, def->metadata(), def->id());
        return (*this)(ret);
    } else {
        return BaseClass::visit(op);
    }
}

Stmt HoistSelectedVar::visit(const If &op) {
    if (toHoist_.count(op->thenCase_->id())) {
        ASSERT(op->thenCase_->nodeType() == ASTNodeType::VarDef);
        auto def = op->thenCase_.as<VarDefNode>();

        Stmt body = def->body_;
        auto name = def->name_;
        if (op->elseCase_.isValid() &&
            allUses(op->elseCase_).count(def->name_)) {
            name = getNewName(def->name_, uni(this->names(), allNames(op)));
            body = renameVar(body, def->name_, name);
        }

        // recurse after making node, to maintain symbol table
        auto ret = makeVarDef(name, def->buffer_, def->viewOf_,
                              makeIf(op->cond_, std::move(body), op->elseCase_,
                                     op->metadata(), op->id()),
                              def->pinned_, def->metadata(), def->id());
        return (*this)(ret);
    } else if (op->elseCase_.isValid() && toHoist_.count(op->elseCase_->id())) {
        ASSERT(op->elseCase_->nodeType() == ASTNodeType::VarDef);
        auto def = op->elseCase_.as<VarDefNode>();

        Stmt body = def->body_;
        auto name = def->name_;
        if (allUses(op->thenCase_).count(def->name_)) {
            name = getNewName(def->name_, uni(this->names(), allNames(op)));
            body = renameVar(body, def->name_, name);
        }

        // recurse after making node, to maintain symbol table
        auto ret = makeVarDef(name, def->buffer_, def->viewOf_,
                              makeIf(op->cond_, op->thenCase_, std::move(body),
                                     op->metadata(), op->id()),
                              def->pinned_, def->metadata(), def->id());
        return (*this)(ret);
    } else {
        return BaseClass::visit(op);
    }
}

Stmt HoistSelectedVar::visit(const Assert &op) {
    if (toHoist_.count(op->body_->id())) {
        ASSERT(op->body_->nodeType() == ASTNodeType::VarDef);
        auto def = op->body_.as<VarDefNode>();
        // recurse after making node, to maintain symbol table
        auto ret = makeVarDef(
            def->name_, def->buffer_, def->viewOf_,
            makeAssert(op->cond_, def->body_, op->metadata(), op->id()),
            def->pinned_, def->metadata(), def->id());
        return (*this)(ret);
    } else {
        return BaseClass::visit(op);
    }
}

Stmt HoistSelectedVar::visit(const Assume &op) {
    if (toHoist_.count(op->body_->id())) {
        ASSERT(op->body_->nodeType() == ASTNodeType::VarDef);
        auto def = op->body_.as<VarDefNode>();
        // recurse after making node, to maintain symbol table
        auto ret = makeVarDef(
            def->name_, def->buffer_, def->viewOf_,
            makeAssert(op->cond_, def->body_, op->metadata(), op->id()),
            def->pinned_, def->metadata(), def->id());
        return (*this)(ret);
    } else {
        return BaseClass::visit(op);
    }
}

Stmt HoistSelectedVar::visit(const StmtSeq &op) {
    for (auto &&[i, stmt] : views::enumerate(op->stmts_)) {
        if (toHoist_.count(stmt->id())) {
            ASSERT(stmt->nodeType() == ASTNodeType::VarDef);
            auto def = stmt.as<VarDefNode>();

            for (auto &&dim : def->buffer_->tensor()->shape()) {
                auto used = allUses(dim);
                for (auto &&[j, other] : views::enumerate(op->stmts_)) {
                    if (j != i && hasIntersect(allWrites(other), used)) {
                        throw InvalidSchedule("Impossible to hoist VarDef " +
                                              toString(def->id()) +
                                              " over StmtSeq " +
                                              toString(op->id()) +
                                              " or its shape will be modified");
                    }
                }
            }

            Stmt body = def->body_;
            auto name = def->name_;
            for (auto &&[j, other] : views::enumerate(op->stmts_)) {
                if (j != i && allUses(other).count(def->name_)) {
                    name = getNewName(def->name_,
                                      uni(this->names(), allNames(op)));
                    body = renameVar(body, def->name_, name);
                    break;
                }
            }

            // recurse after making node, to maintain symbol table
            std::vector<Stmt> stmts = op->stmts_;
            stmts[i] = std::move(body);
            auto ret = makeVarDef(
                name, def->buffer_, def->viewOf_,
                makeStmtSeq(std::move(stmts), op->metadata(), op->id()),
                def->pinned_, def->metadata(), def->id());
            return (*this)(ret);
        }
    }
    return BaseClass::visit(op);
}

Stmt hoistSelectedVar(const Stmt &_op, const Ref<Selector> &_selector) {
    auto selector = Ref<BothSelector>::make(
        Ref<NodeTypeSelector>::make(ASTNodeType::VarDef), _selector);
    auto op = _op;
    while (true) {
        auto nodes = findAllStmt(op, selector);
        if (nodes.empty()) {
            break;
        }

        auto ids = ranges::to<std::unordered_set>(
            nodes |
            views::transform([](const auto &node) { return node->id(); }));
        op = HoistSelectedVar(ids)(op);
    }
    return op;
}

Stmt hoistSelectedVar(const Stmt &op, const std::string &selector) {
    return hoistSelectedVar(op, parseSelector(selector));
}

} // namespace freetensor
