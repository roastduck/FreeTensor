#include <functional>

#include <selector.h>

#include <selector_lexer.h>
#include <selector_parser.h>

namespace freetensor {

bool NotSelector::matchImpl(const Stmt &stmt) { return !sub_->match(stmt); }

bool BothSelector::matchImpl(const Stmt &stmt) {
    return lhs_->match(stmt) && rhs_->match(stmt);
}

bool EitherSelector::matchImpl(const Stmt &stmt) {
    return lhs_->match(stmt) || rhs_->match(stmt);
}

bool NodeTypeSelector::matchImpl(const Stmt &stmt) {
    return stmt->nodeType() == nodeType_;
}

bool ChildSelector::matchImpl(const Stmt &stmt) {
    auto p = stmt->parentStmt();
    return p.isValid() && parent_->match(p);
}

bool DescendantSelector::matchImpl(const Stmt &_stmt) {
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

bool ParentSelector::matchImpl(const Stmt &stmt) {
    for (auto &&c : stmt->children()) {
        if (child_->match(c)) {
            return true;
        }
    }
    return false;
}

bool AncestorSelector::matchImpl(const Stmt &stmt) {
    std::function<bool(const Stmt &s)> recur = [&](const Stmt &s) {
        if (descendant_->match(s)) {
            return true;
        }
        if (middle_.isValid() && !middle_->match(s)) {
            return false;
        }
        for (auto &&c : s->children()) {
            if (recur(c)) {
                return true;
            }
        }
        return false;
    };
    for (auto &&c : stmt->children()) {
        if (recur(c)) {
            return true;
        }
    }
    return false;
}

bool RootNodeSelector::matchImpl(const Stmt &stmt) {
    return !stmt->parentStmt().isValid();
}

bool LeafNodeSelector::matchImpl(const Stmt &stmt) {
    return stmt->children().empty();
}

bool NotMetadataSelector::matchImpl(const Metadata &md) {
    return !sub_->match(md);
}

bool BothMetadataSelector::matchImpl(const Metadata &md) {
    return lhs_->match(md) && rhs_->match(md);
}

bool EitherMetadataSelector::matchImpl(const Metadata &md) {
    return lhs_->match(md) || rhs_->match(md);
}

bool IDSelector::matchImpl(const Stmt &stmt) { return stmt->id() == id_; }
bool IDSelector::matchImpl(const Metadata &md) {
    return md->getType() == MetadataType::Anonymous &&
           md.as<AnonymousMetadataContent>()->id() == id_;
}

bool LabelSelector::matchImpl(const Metadata &md) {
    if (md->getType() != MetadataType::Source)
        return false;
    return md.as<SourceMetadataContent>()->labelsSet().count(label_);
}

bool TransformedSelector::matchImpl(const Metadata &_md) {
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

bool DirectCallerSelector::matchImpl(const Metadata &_md) {
    if (_md->getType() != MetadataType::Source)
        return false;
    auto md = _md.as<SourceMetadataContent>();
    if (!md->caller().isValid())
        return false;
    return caller_->match(md->caller());
}

bool CallerSelector::matchImpl(const Metadata &_md) {
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

bool RootCallSelector::matchImpl(const Metadata &md) {
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
