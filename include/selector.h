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

class NotSelector : public Selector {
    Ref<Selector> sub_;

  public:
    NotSelector(const Ref<Selector> &sub) : sub_(sub) {}
    bool match(const Stmt &stmt) const override;
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
    Ref<Selector> parent_;

  public:
    ChildSelector(const Ref<Selector> &parent) : parent_(parent) {}
    bool match(const Stmt &stmt) const override;
};

class DescendantSelector : public Selector {
    Ref<Selector> ancestor_, middle_;

  public:
    DescendantSelector(const Ref<Selector> &ancestor,
                       const Ref<Selector> &middle = nullptr)
        : ancestor_(ancestor), middle_(middle) {}
    bool match(const Stmt &stmt) const override;
};

class LeafSelector : public Selector {
  public:
    virtual bool match(const Metadata &md) const = 0;
    virtual bool match(const Stmt &stmt) const override {
        return stmt->metadata().isValid() && match(stmt->metadata());
    }
};

class NotLeafSelector : public LeafSelector {
    Ref<LeafSelector> sub_;

  public:
    NotLeafSelector(const Ref<LeafSelector> &sub) : sub_(sub) {}
    bool match(const Metadata &md) const override;
};

class BothLeafSelector : public LeafSelector {
    Ref<LeafSelector> lhs_, rhs_;

  public:
    BothLeafSelector(const Ref<LeafSelector> &lhs, const Ref<LeafSelector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
    bool match(const Metadata &md) const override;
};

class EitherLeafSelector : public LeafSelector {
    Ref<LeafSelector> lhs_, rhs_;

  public:
    EitherLeafSelector(const Ref<LeafSelector> &lhs,
                       const Ref<LeafSelector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
    bool match(const Metadata &md) const override;
};

class IDSelector : public LeafSelector {
    ID id_;

  public:
    IDSelector(const ID &id) : id_(id) {}
    bool match(const Metadata &md) const override;
    bool match(const Stmt &stmt) const override;
};

class LabelSelector : public LeafSelector {
    std::string label_;

  public:
    LabelSelector(const std::string &label) : label_(label) {}
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

class DirectCallerSelector : public LeafSelector {
    Ref<LeafSelector> caller_;

  public:
    DirectCallerSelector(const Ref<LeafSelector> &caller) : caller_(caller) {}
    bool match(const Metadata &md) const override;
};

class CallerSelector : public LeafSelector {
    Ref<LeafSelector> caller_, middle_;

  public:
    CallerSelector(const Ref<LeafSelector> &caller,
                   const Ref<LeafSelector> &middle = nullptr)
        : caller_(caller), middle_(middle) {}
    bool match(const Metadata &md) const override;
};

Ref<Selector> parseSelector(const std::string &str);

} // namespace freetensor

#endif // FREE_TENSOR_SELECTOR_H
