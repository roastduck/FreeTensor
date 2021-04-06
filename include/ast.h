#ifndef AST_H
#define AST_H

#include <ref.h>

namespace ir {

enum class ASTNodeType : int {
    Any,
    AnyExpr,
    StmtSeq,
    VarDef,
    Var,
    Store,
    Load,
    ReduceTo,
    IntConst,
    FloatConst,
    BoolConst,
    Add,
    Sub,
    Mul,
    RealDiv,
    FloorDiv,
    CeilDiv,
    RoundTowards0Div,
    Mod,
    Min,
    Max,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NE,
    LAnd,
    LOr,
    LNot,
    For,
    If,
    Assert,
    Intrinsic,
    Eval,
};

std::string toString(ASTNodeType type);

#define DEFINE_NODE_ACCESS(name)                                               \
  protected:                                                                   \
    name##Node() = default; /* Must be constructed in Ref */                   \
                                                                               \
    friend class Allocator<name##Node>;

#define DEFINE_NODE_TRAIT(name)                                                \
    DEFINE_NODE_ACCESS(name)                                                   \
  public:                                                                      \
    virtual ASTNodeType nodeType() const override { return ASTNodeType::name; }

class ASTNode {
    friend class Disambiguous;

    bool isSubTree_ = false;

  public:
    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

    void setIsSubTree(bool isSubTree = true) { isSubTree_ = isSubTree; }
    bool isSubTree() const { return isSubTree_; }

    DEFINE_NODE_ACCESS(AST);
};
typedef Ref<ASTNode> AST;

class ExprNode : public ASTNode {
    DEFINE_NODE_ACCESS(Expr);
};
typedef Ref<ExprNode> Expr;

class StmtNode : public ASTNode {
    std::string id_;
    static uint64_t idCnt_;

  public:
    void setId(const std::string &id);
    const std::string &id() const;
    bool hasNamedId() const;

    DEFINE_NODE_ACCESS(Stmt);
};
typedef Ref<StmtNode> Stmt;

Expr deepCopy(const Expr &op);
Stmt deepCopy(const Stmt &op);

enum NullPolicy : int { NotNull, Nullable };

/**
 * To ensure there will not be two nodes in one AST sharing the same address
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTree {
    Ref<T> obj_;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTree() {}
    SubTree(std::nullptr_t) { ASSERT(POLICY == NullPolicy::Nullable); }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(const Ref<U> &obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTree(Ref<U> &&obj) : obj_(obj) {
        if (obj_.isValid()) {
            if (obj_->isSubTree()) {
                obj_ = deepCopy(obj).template as<T>();
            }
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }

    template <NullPolicy OTHER_POLICY>
    SubTree(SubTree<T, OTHER_POLICY> &&other) : obj_(std::move(other.obj_)) {
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
    }

    template <NullPolicy OTHER_POLICY>
    explicit SubTree(const SubTree<T, OTHER_POLICY> &other) {
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
    }

    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(SubTree<T, OTHER_POLICY> &&other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        obj_ = std::move(other.obj_);
        if (!obj_.isValid()) {
            ASSERT(POLICY == NullPolicy::Nullable);
        }
        return *this;
    }

    template <NullPolicy OTHER_POLICY>
    SubTree &operator=(const SubTree<T, OTHER_POLICY> &other) {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
        if (other.obj_.isValid()) {
            obj_ = deepCopy(other.obj_).template as<T>();
            ASSERT(!obj_->isSubTree());
            obj_->setIsSubTree();
        } else {
            ASSERT(POLICY == NullPolicy::Nullable);
            obj_ = nullptr;
        }
        return *this;
    }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
    operator Ref<U>() const {
        return obj_;
    }

    T &operator*() const { return *obj_; }
    T *operator->() const { return obj_.get(); }

    template <class U> Ref<U> as() const { return obj_.template as<U>(); }

    bool isValid() const { return obj_.isValid(); }
};

} // namespace ir

#endif // AST_H
