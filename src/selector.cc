#include <selector.h>

#include <selector_lexer.h>
#include <selector_parser.h>

namespace freetensor {

bool NotSelector::match(const Stmt &stmt) const { return !sub_->match(stmt); }

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
        if (middle_.isValid() && !middle_->match(stmt)) {
            return false;
        }
    }
    return false;
}

bool RootNodeSelector::match(const Stmt &stmt) const {
    return !stmt->parentStmt().isValid();
}

bool NotLeafSelector::match(const Metadata &md) const {
    return !sub_->match(md);
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
    if (md->getType() != MetadataType::Source)
        return false;
    return md.as<SourceMetadataContent>()->labelsSet().count(label_);
}

bool TransformedSelector::match(const Metadata &_md) const {
    if (_md->getType() != MetadataType::Transformed)
        return false;
    auto md = _md.as<TransformedMetadataContent>();
    if (md->op() != op_ || sources_.size() != md->sources().size())
        return false;
    for (auto &&[sel, md] : views::zip(sources_, md->sources()))
        if (!sel->match(md))
            return false;
    return true;
}

bool DirectCallerSelector::match(const Metadata &_md) const {
    if (_md->getType() != MetadataType::Source)
        return false;
    auto md = _md.as<SourceMetadataContent>();
    if (!md->caller().isValid())
        return false;
    return caller_->match(md->caller());
}

bool CallerSelector::match(const Metadata &_md) const {
    if (_md->getType() != MetadataType::Source)
        return false;
    for (auto md = _md.as<SourceMetadataContent>()->caller(); md.isValid();) {
        if (caller_->match(md)) {
            return true;
        }
        if (middle_.isValid() && !middle_->match(md)) {
            return false;
        }
        if (md->getType() == MetadataType::Source) {
            md = md.as<SourceMetadataContent>()->caller();
        }
    }
    return false;
}

bool RootCallSelector::match(const Metadata &md) const {
    if (md->getType() != MetadataType::Source)
        return false;
    return !md.as<SourceMetadataContent>()->caller().isValid();
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
