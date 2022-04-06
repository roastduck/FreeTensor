#ifndef AST_H
#define AST_H

#include <atomic>
#include <string>

#include <ref.h>
#include <sub_tree.h>

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

class ASTNode : public ASTPart {
  protected:
    size_t hash_ = ~0ull;

  public:
#ifdef IR_DEBUG_LOG_NODE
    std::string debugCreator_ = "Python API";
#endif

    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

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

} // namespace ir

namespace std {

template <> struct hash<ir::ID> { size_t operator()(const ir::ID &id) const; };

} // namespace std

#endif // AST_H
