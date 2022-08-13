#include <selector.h>

#include <selector_lexer.h>
#include <selector_parser.h>

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

bool IDSelector::match(const Stmt &stmt) const { return stmt->id() == id_; }

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

Ref<Selector> parseSelector(const std::string &str) {
    try {
        antlr4::ANTLRInputStream charStream(str);
        selector_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        selector_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        return parser.selector()->s;
    } catch (const antlr4::ParseCancellationException &e) {
        throw ParserError((std::string) "Parser error: " + e.what());
    }
}

} // namespace freetensor
