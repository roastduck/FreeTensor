#ifndef FREE_TENSOR_SELECTOR_H
#define FREE_TENSOR_SELECTOR_H

#include <string>

#include <ast.h>
#include <stmt.h>

namespace freetensor {

class Selector {
  public:
    virtual ~Selector() {}
    virtual bool match(const Stmt &stmt) const = 0;
};

class BothSelector : public Selector {
    Ref<Selector> lhs_, rhs_;

  public:
    BothSelector(const Ref<Selector> &lhs, const Ref<Selector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
    bool match(const Stmt &stmt) const override;
};

class EitherSelector : public Selector {
    Ref<Selector> lhs_, rhs_;

  public:
    EitherSelector(const Ref<Selector> &lhs, const Ref<Selector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
    bool match(const Stmt &stmt) const override;
};

class NodeTypeSelector : public Selector {
    ASTNodeType nodeType_;

  public:
    NodeTypeSelector(const ASTNodeType &nodeType) : nodeType_(nodeType) {}
    bool match(const Stmt &stmt) const override;
};

class ChildSelector : public Selector {
    Ref<Selector> parent_, child_;

  public:
    ChildSelector(const Ref<Selector> &parent, const Ref<Selector> &child)
        : parent_(parent), child_(child) {}
    bool match(const Stmt &stmt) const override;
};

class DescendantSelector : public Selector {
    Ref<Selector> ancestor_, descendant_;

  public:
    DescendantSelector(const Ref<Selector> &ancestor,
                       const Ref<Selector> &descendant)
        : ancestor_(ancestor), descendant_(descendant) {}
    bool match(const Stmt &stmt) const override;
};

class LeafSelector : public Selector {
  public:
    virtual bool match(const Metadata &md) const = 0;
    virtual bool match(const Stmt &stmt) const override {
        return stmt->metadata().isValid() && match(stmt->metadata());
    }
};

class IDSelector : public LeafSelector {
    ID id_;

  public:
    IDSelector(const ID &id) : id_(id) {}
    bool match(const Metadata &md) const override;
    bool match(const Stmt &stmt) const override;
};

class LabelSelector : public LeafSelector {
    std::vector<std::string> labels_;

  public:
    LabelSelector(const std::vector<std::string> &label) : labels_(label) {}
    bool match(const Metadata &md) const override;
};

class TransformedSelector : public LeafSelector {
    std::string op_;
    std::vector<Ref<LeafSelector>> sources_;

  public:
    TransformedSelector(const std::string &op,
                        const std::vector<Ref<LeafSelector>> sources)
        : op_(op), sources_(sources) {}
    bool match(const Metadata &md) const override;
};

class CallerSelector : public LeafSelector {
    Ref<LeafSelector> self_, caller_;

  public:
    CallerSelector(const Ref<LeafSelector> &self,
                   const Ref<LeafSelector> &caller)
        : self_(self), caller_(caller) {}
    bool match(const Metadata &md) const override;
};

Ref<Selector> parseSelector(const std::string &str);

} // namespace freetensor

#endif // FREE_TENSOR_SELECTOR_H
