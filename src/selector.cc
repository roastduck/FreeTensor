#include <selector.h>

namespace freetensor {

bool BothSelector::match(const Stmt &stmt) const {
    return lhs_->match(stmt) && rhs_->match(stmt);
}

bool EitherSelector::match(const Stmt &stmt) const {
    return lhs_->match(stmt) || rhs_->match(stmt);
}

bool LabelSelector::match(const Stmt &stmt) const {
    if (stmt->metadata()->getType() != MetadataType::Source)
        return false;
    const auto &labels = stmt->metadata().as<SourceMetadataContent>()->labels();
    return std::find(labels.begin(), labels.end(), label_) != labels.end();
}

bool NodeTypeSelector::match(const Stmt &stmt) const {
    return stmt->nodeType() == nodeType_;
}

bool ChildSelector::match(const Stmt &stmt) const {
    return child_->match(stmt) && stmt->parent()->isAST() &&
           stmt->parent().as<ASTNode>()->isStmt() &&
           parent_->match(stmt->parent().as<StmtNode>());
}

bool DescendantSelector::match(const Stmt &_stmt) const {
    auto stmt = _stmt;
    if (!descendant_->match(stmt))
        return false;
    while (stmt->parent()->isAST() && stmt->parent().as<ASTNode>()->isStmt()) {
        stmt = stmt->parent().as<StmtNode>();
        if (ancestor_->match(stmt))
            return true;
    }
    return false;
}

} // namespace freetensor
