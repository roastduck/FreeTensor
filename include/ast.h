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

#define DEFINE_NODE_ACCESS(name) DEFINE_AST_PART_ACCESS(name##Node)

#define DEFINE_NODE_TRAIT(name)                                                \
    DEFINE_NODE_ACCESS(name)                                                   \
  public:                                                                      \
    virtual ASTNodeType nodeType() const override { return ASTNodeType::name; }

/**
 * Base class of all nodes in an AST
 *
 * `ASTNode` is the minimal unit that can be traversed by a `Visitor`, or
 * traversed and modified by a `Mutator`. An `ASTNode` is a derived class of
 * `ASTPart`, and a derived node of `ASTNode` may contain other `ASTPart`s
 */
class ASTNode : public ASTPart {
  public:
#ifdef IR_DEBUG_LOG_NODE
    std::string debugCreator_ = "Python API";
#endif

    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

    bool isAST() const override { return true; }
    virtual bool isFunc() const { return false; }
    virtual bool isStmt() const { return false; }
    virtual bool isExpr() const { return false; }

    Ref<ASTNode> parentAST() const;

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

/**
 * Base class of all expression nodes in an AST
 */
class ExprNode : public ASTNode {
  public:
    bool isExpr() const override { return true; }

    virtual bool isConst() const { return false; }
    virtual bool isBinary() const { return false; }
    virtual bool isUnary() const { return false; }

    Ref<ExprNode> parentExpr() const;

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

/**
 * Base class of all statement nodes in an AST
 */
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

    Ref<StmtNode> parentStmt() const;
    Ref<StmtNode> parentCtrlFlow() const;
    Ref<StmtNode> prevStmt() const;
    Ref<StmtNode> nextStmt() const;

    /**
     * Find an ancestor by ID. `this` itself is also considered
     */
    Ref<StmtNode> ancestorById(const ID &lookup) const;

    bool isAncestorOf(const Stmt &other) const;
    bool isBefore(const Stmt &other) const;

    DEFINE_NODE_ACCESS(Stmt);
};

Expr deepCopy(const Expr &op);
Stmt deepCopy(const Stmt &op);

AST lcaAST(const AST &lhs, const AST &rhs);
Expr lcaExpr(const Expr &lhs, const Expr &rhs);
Stmt lcaStmt(const Stmt &lhs, const Stmt &rhs);

} // namespace ir

namespace std {

template <> struct hash<ir::ID> { size_t operator()(const ir::ID &id) const; };

} // namespace std

#endif // AST_H
