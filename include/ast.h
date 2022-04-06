#ifndef AST_H
#define AST_H

#include <atomic>
#include <string>
#include <vector>

#include <ref.h>

namespace ir {

enum class ASTNodeType : int {
    // Matching
    Any,
    AnyExpr,

    // Function
    Func,

    // Memory Access
    Store,
    ReduceTo,
    Load,

    // Structral statements
    StmtSeq,
    VarDef,
    For,
    If,
    Assert,
    Assume,

    // Calls to external libs
    MatMul,

    // Other statements
    Eval,

    // Values
    Var,
    IntConst,
    FloatConst,
    BoolConst,

    // Binary ops
    Add,
    Sub,
    Mul,
    RealDiv,
    FloorDiv,
    CeilDiv,
    RoundTowards0Div,
    Mod,
    Remainder,
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

    // Unary ops
    LNot,
    Sqrt,
    Exp,
    Square,
    Sigmoid,
    Tanh,
    Abs,
    Floor,
    Ceil,

    // Other expressions
    IfExpr,
    Cast,
    Intrinsic,
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

  protected:
    size_t hash_ = ~0ull;

  public:
#ifdef IR_DEBUG_LOG_NODE
    std::string debugCreator_ = "Python API";
#endif

    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

    void setIsSubTree(bool isSubTree = true) { isSubTree_ = isSubTree; }
    bool isSubTree() const { return isSubTree_; }

    virtual bool isFunc() const { return false; }
    virtual bool isStmt() const { return false; }
    virtual bool isExpr() const { return false; }

    size_t hash();
    virtual void compHash() = 0;

    DEFINE_NODE_ACCESS(AST);
};
typedef Ref<ASTNode> AST;

#ifdef IR_DEBUG_LOG_NODE
#define makeNode(type, ...)                                                    \
    ({                                                                         \
        auto __x = _make##type(__VA_ARGS__);                                   \
        __x->debugCreator_ = __FILE__ ":" + std::to_string(__LINE__);          \
        __x;                                                                   \
    }) // GCC Extension: Statement expression
#define COPY_DEBUG_INFO(ret, old)                                              \
    ({                                                                         \
        auto __x = (ret);                                                      \
        auto __y = (old);                                                      \
        __x->debugCreator_ = __y->debugCreator_;                               \
        __x;                                                                   \
    }) // GCC Extension: Statement expression
#else
#define makeNode(type, ...) _make##type(__VA_ARGS__)
#define COPY_DEBUG_INFO(ret, old) (ret)
#endif

class ExprNode : public ASTNode {
  public:
    bool isExpr() const override { return true; }

    virtual bool isConst() const { return false; }
    virtual bool isBinary() const { return false; }
    virtual bool isUnary() const { return false; }

    DEFINE_NODE_ACCESS(Expr);
};
typedef Ref<ExprNode> Expr;

class StmtNode;
typedef Ref<StmtNode> Stmt;

/**
 * Identify an Stmt or Expr acrossing passes, so we do not need to pass pointers
 *
 * An Stmt is identified by a string-typed id_ property, which is unique to each
 * Stmt node
 *
 * An Expr is identified by the hash of itself, combined with the string-typed
 * ID of the Stmt its in, so that any identical Expr in the same Stmt is treated
 * as the same node
 */
class ID {
    friend StmtNode;

    std::string stmtId_;
    Expr expr_; /// null for Stmt

  public:
    ID() {}
    ID(const char *stmtId) : stmtId_(stmtId) {}
    ID(const std::string &stmtId) : stmtId_(stmtId) {}
    explicit ID(const Stmt &stmt);

    template <class T> ID(const Expr &expr, T &&parent) : ID(parent) {
        expr_ = expr;
    }

    bool isValid() const { return !stmtId_.empty(); }

    const std::string &strId() const;

    friend std::string toString(const ID &id);
    friend bool operator==(const ID &lhs, const ID &rhs);
    friend bool operator!=(const ID &lhs, const ID &rhs);
    friend struct ::std::hash<ID>;
};

std::string toString(const ID &id);

class StmtNode : public ASTNode {
    friend ID;

    std::string id_;
    static std::atomic<uint64_t> idCnt_;

  public:
    static std::string newId();

    void setId(const ID &id);
    ID id() const;
    bool hasNamedId() const;

    bool isStmt() const override { return true; }

    DEFINE_NODE_ACCESS(Stmt);
};

Expr deepCopy(const Expr &op);
Stmt deepCopy(const Stmt &op);

enum NullPolicy : int { NotNull, Nullable };

/**
 * Plugging a Ref as a sub-tree in the AST
 *
 * This class ensures that each Ref of an AST node having a single parent. In
 * other words, there will not be two nodes in one AST sharing the same address.
 * If an AST node is assigned as a SubTree, but it has already been in another
 * SubTree, it will be automatically copied
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTree {
    Ref<T> obj_;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTree() {}
    SubTree(std::nullptr_t) { ASSERT(POLICY == NullPolicy::Nullable); }

    ~SubTree() {
        if (obj_.isValid()) {
            obj_->setIsSubTree(false);
        }
    }

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

/**
 * A list of SubTree
 *
 * This class can be used as a list of its inner AST nodes
 */
template <class T, NullPolicy POLICY = NullPolicy::NotNull> class SubTreeList {
    std::vector<SubTree<T, POLICY>> objs_;

    template <class, NullPolicy> friend class SubTree;

  public:
    SubTreeList() {}

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(const std::vector<Ref<U>> &objs)
        : objs_(objs.begin(), objs.end()) {}
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::vector<Ref<U>> &&objs) {
        objs_.reserve(objs.size());
        for (auto &obj : objs) {
            objs_.emplace_back(std::move(obj));
        }
    }
    template <class U,
              typename std::enable_if_t<std::is_base_of_v<T, U>> * = nullptr>
    SubTreeList(std::initializer_list<Ref<U>> objs) {
        objs_.reserve(objs.size());
        for (auto &obj : objs) {
            objs_.emplace_back(std::move(obj));
        }
    }

    SubTreeList(SubTreeList<T, POLICY> &&other)
        : objs_(std::move(other.objs_)) {}
    explicit SubTreeList(const SubTreeList<T, POLICY> &other)
        : objs_(other.objs_) {}

    template <NullPolicy OTHER_POLICY>
    SubTreeList(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.reserve(other.objs_.size());
        for (auto &obj : other.objs_) {
            objs_.emplace_back(std::move(obj));
        }
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList(const SubTreeList<T, OTHER_POLICY> &other)
        : objs_(other.objs_.begin(), other.objs_.end()) {}

    SubTreeList &operator=(SubTreeList<T, POLICY> &&other) {
        objs_ = std::move(other.objs_);
        return *this;
    }
    SubTreeList &operator=(const SubTreeList<T, POLICY> &other) {
        objs_ = other.objs_;
        return *this;
    }

    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(SubTreeList<T, OTHER_POLICY> &&other) {
        objs_.clear();
        objs_.reserve(other.objs_.size());
        for (auto &obj : other.objs_) {
            objs_.emplace_back(std::move(obj));
        }
        return *this;
    }
    template <NullPolicy OTHER_POLICY>
    SubTreeList &operator=(const SubTreeList<T, OTHER_POLICY> &other) {
        objs_ = std::vector<SubTree<T, POLICY>>(other.objs_.begin(),
                                                other.objs_.end());
        return *this;
    }

    template <class U,
              typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
    operator std::vector<Ref<U>>() const {
        return std::vector<Ref<U>>(objs_.begin(), objs_.end());
    }

    auto begin() { return objs_.begin(); }
    auto begin() const { return objs_.begin(); }
    auto end() { return objs_.end(); }
    auto end() const { return objs_.end(); }
    auto rbegin() { return objs_.rbegin(); }
    auto rbegin() const { return objs_.rbegin(); }
    auto rend() { return objs_.rend(); }
    auto rend() const { return objs_.rend(); }
    auto size() const { return objs_.size(); }
    auto empty() const { return objs_.empty(); }
    template <class U> auto &operator[](U &&i) {
        return objs_[std::forward<U>(i)];
    }
    template <class U> const auto &operator[](U &&i) const {
        return objs_[std::forward<U>(i)];
    }
    template <class U> auto &at(U &&i) { return objs_.at(std::forward<U>(i)); }
    template <class U> const auto &at(U &&i) const {
        return objs_.at(std::forward<U>(i));
    }
    auto &front() { return objs_.front(); }
    const auto &front() const { return objs_.front(); }
    auto &back() { return objs_.back(); }
    const auto &back() const { return objs_.back(); }
    template <class U> void emplace_back(U &&x) {
        objs_.emplace_back(std::forward<U>(x));
    }
    template <class U> void push_back(U &&x) {
        objs_.push_back(std::forward<U>(x));
    }
    template <class... U> auto insert(U &&...i) {
        return objs_.insert(std::forward<U>(i)...);
    }
    template <class U> auto erase(U &&i) {
        return objs_.erase(std::forward<U>(i));
    }
    void pop_back() { objs_.pop_back(); }
};

} // namespace ir

namespace std {

template <> struct hash<ir::ID> { size_t operator()(const ir::ID &id) const; };

} // namespace std

#endif // AST_H
