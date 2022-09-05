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

bool NodeTypeSelector::match(const Stmt &stmt) const {
    return stmt->nodeType() == nodeType_;
}

bool ChildSelector::match(const Stmt &stmt) const {
    auto p = stmt->parentStmt();
    return p.isValid() && parent_->match(p);
}

bool DescendantSelector::match(const Stmt &_stmt) const {
    for (auto stmt = _stmt->parentStmt(); stmt.isValid();
         stmt = stmt->parentStmt()) {
        if (ancestor_->match(stmt))
            return true;
    }
    return false;
}

bool BothLeafSelector::match(const Metadata &md) const {
    return lhs_->match(md) && rhs_->match(md);
}

bool EitherLeafSelector::match(const Metadata &md) const {
    return lhs_->match(md) || rhs_->match(md);
}

bool IDSelector::match(const Stmt &stmt) const { return stmt->id() == id_; }
bool IDSelector::match(const Metadata &md) const {
    return md->getType() == MetadataType::Anonymous &&
           md.as<AnonymousMetadataContent>()->id() == id_;
}

bool LabelSelector::match(const Metadata &md) const {
    if (!md.isValid() || md->getType() != MetadataType::Source)
        return false;
    const auto &labelsSet = md.as<SourceMetadataContent>()->labelsSet();
    for (const auto &l : labels_)
        if (labelsSet.count(l) == 0)
            return false;
    return true;
}

bool TransformedSelector::match(const Metadata &_md) const {
    if (_md->getType() != MetadataType::Transformed)
        return false;
    auto md = _md.as<TransformedMetadataContent>();
    if (md->op() != op_ || sources_.size() != md->sources().size())
        return false;
    for (auto &&[sel, md] : iter::zip(sources_, md->sources()))
        if (!sel->match(md))
            return false;
    return true;
}

bool CallerSelector::match(const Metadata &_md) const {
    if (_md->getType() != MetadataType::Source)
        return false;
    auto md = _md.as<SourceMetadataContent>();
    if (!md->caller().isValid())
        return false;
    return caller_->match(md->caller());
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
