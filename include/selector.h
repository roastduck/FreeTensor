#ifndef FREE_TENSOR_SELECTOR_H
#define FREE_TENSOR_SELECTOR_H

#include <string>
#include <unordered_map>

#include <ast.h>
#include <stmt.h>

namespace freetensor {

class Selector {
    std::unordered_map<Stmt, bool> cache_;

  protected:
    virtual bool matchImpl(const Stmt &stmt) = 0;

  public:
    virtual ~Selector() {}

    bool match(const Stmt &stmt) {
        if (auto it = cache_.find(stmt); it != cache_.end()) {
            return it->second;
        }
        return cache_[stmt] = matchImpl(stmt);
    }
};

class NotSelector : public Selector {
    Ref<Selector> sub_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    NotSelector(const Ref<Selector> &sub) : sub_(sub) {}
};

class BothSelector : public Selector {
    Ref<Selector> lhs_, rhs_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    BothSelector(const Ref<Selector> &lhs, const Ref<Selector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
};

class EitherSelector : public Selector {
    Ref<Selector> lhs_, rhs_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    EitherSelector(const Ref<Selector> &lhs, const Ref<Selector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
};

class NodeTypeSelector : public Selector {
    ASTNodeType nodeType_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    NodeTypeSelector(const ASTNodeType &nodeType) : nodeType_(nodeType) {}
};

class ChildSelector : public Selector {
    Ref<Selector> parent_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    ChildSelector(const Ref<Selector> &parent) : parent_(parent) {}
};

class DescendantSelector : public Selector {
    Ref<Selector> ancestor_, middle_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    DescendantSelector(const Ref<Selector> &ancestor,
                       const Ref<Selector> &middle = nullptr)
        : ancestor_(ancestor), middle_(middle) {}
};

class ParentSelector : public Selector {
    Ref<Selector> child_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    ParentSelector(const Ref<Selector> &child) : child_(child) {}
};

class AncestorSelector : public Selector {
    Ref<Selector> descendant_, middle_;

  protected:
    bool matchImpl(const Stmt &stmt) override;

  public:
    AncestorSelector(const Ref<Selector> &descendant,
                     const Ref<Selector> &middle = nullptr)
        : descendant_(descendant), middle_(middle) {}
};

class RootNodeSelector : public Selector {
  protected:
    bool matchImpl(const Stmt &stmt) override;
};

class LeafSelector : public Selector {
  protected:
    virtual bool matchImpl(const Metadata &md) = 0;
    virtual bool matchImpl(const Stmt &stmt) override {
        return stmt->metadata().isValid() && match(stmt->metadata());
    }

  public:
    bool match(const Metadata &md) {
        // TODO: Memoize the result
        return matchImpl(md);
    }
};

class NotLeafSelector : public LeafSelector {
    Ref<LeafSelector> sub_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    NotLeafSelector(const Ref<LeafSelector> &sub) : sub_(sub) {}
};

class BothLeafSelector : public LeafSelector {
    Ref<LeafSelector> lhs_, rhs_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    BothLeafSelector(const Ref<LeafSelector> &lhs, const Ref<LeafSelector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
};

class EitherLeafSelector : public LeafSelector {
    Ref<LeafSelector> lhs_, rhs_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    EitherLeafSelector(const Ref<LeafSelector> &lhs,
                       const Ref<LeafSelector> &rhs)
        : lhs_(lhs), rhs_(rhs) {}
};

class IDSelector : public LeafSelector {
    ID id_;

  protected:
    bool matchImpl(const Metadata &md) override;
    bool matchImpl(const Stmt &stmt) override;

  public:
    IDSelector(const ID &id) : id_(id) {}
};

class LabelSelector : public LeafSelector {
    std::string label_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    LabelSelector(const std::string &label) : label_(label) {}
};

class TransformedSelector : public LeafSelector {
    std::string op_;
    std::vector<Ref<LeafSelector>> sources_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    TransformedSelector(const std::string &op,
                        const std::vector<Ref<LeafSelector>> sources)
        : op_(op), sources_(sources) {}
};

class DirectCallerSelector : public LeafSelector {
    Ref<LeafSelector> caller_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    DirectCallerSelector(const Ref<LeafSelector> &caller) : caller_(caller) {}
};

class CallerSelector : public LeafSelector {
    Ref<LeafSelector> caller_, middle_;

  protected:
    bool matchImpl(const Metadata &md) override;

  public:
    CallerSelector(const Ref<LeafSelector> &caller,
                   const Ref<LeafSelector> &middle = nullptr)
        : caller_(caller), middle_(middle) {}
};

class RootCallSelector : public LeafSelector {
  protected:
    bool matchImpl(const Metadata &md) override;
};

Ref<Selector> parseSelector(const std::string &str);

} // namespace freetensor

#endif // FREE_TENSOR_SELECTOR_H
