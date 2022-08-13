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

class LabelSelector : public Selector {
    std::string label_;

  public:
    LabelSelector(const std::string &label) : label_(label) {}
    bool match(const Stmt &stmt) const override;
};

class IDSelector : public Selector {
    ID id_;

  public:
    IDSelector(const ID &id) : id_(id) {}
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

Ref<Selector> parseSelector(std::string str);

} // namespace freetensor

#endif // FREE_TENSOR_SELECTOR_H
