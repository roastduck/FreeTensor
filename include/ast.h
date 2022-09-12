#ifndef FREE_TENSOR_AST_H
#define FREE_TENSOR_AST_H

#include <atomic>
#include <functional>
#include <iostream>
#include <string>

#include <data_type.h>
#include <id.h>
#include <metadata.h>
#include <ref.h>
#include <serialize/to_string.h>
#include <sub_tree.h>

namespace freetensor {

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
    Alloc,
    Free,

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

std::ostream &operator<<(std::ostream &os, ASTNodeType type);

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
#ifdef FT_DEBUG_LOG_NODE
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

#ifdef FT_DEBUG_LOG_NODE
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
  protected:
    DataType dtype_ = DataType::Invalid;

  public:
    bool isExpr() const override { return true; }

    virtual bool isConst() const { return false; }
    virtual bool isBinary() const { return false; }
    virtual bool isUnary() const { return false; }

    Ref<ExprNode> parentExpr() const;

    void modifiedHook() override {
        ASTNode::modifiedHook();
        resetDType();
    }

    DataType dtype();
    void resetDType();
    virtual void inferDType() = 0;

    DEFINE_NODE_ACCESS(Expr);
};
typedef Ref<ExprNode> Expr;

class StmtNode;
typedef Ref<StmtNode> Stmt;

/**
 * Identify an Stmt or Expr acrossing passes, so we do not need to pass pointers
 *
 * An Expr is identified by the hash of itself, combined with the string-typed
 * ID of the Stmt its in, so that any identical Expr in the same Stmt is treated
 * as the same node
 */
class StmtOrExprID {
    ID stmtId_;
    Expr expr_;

  public:
    StmtOrExprID(const ID &stmtId) : stmtId_(stmtId) {}

    StmtOrExprID(const Expr &expr, const ID &stmtId)
        : stmtId_(stmtId), expr_(expr) {}

    template <std::convertible_to<Stmt> T>
    StmtOrExprID(const Expr &expr, T &&parent) : stmtId_(parent->id()) {
        expr_ = expr;
    }

    friend std::ostream &operator<<(std::ostream &os, const StmtOrExprID &id);
    friend bool operator==(const StmtOrExprID &lhs, const StmtOrExprID &rhs);
    friend struct ::std::hash<StmtOrExprID>;
};

/**
 * Base class of all statement nodes in an AST
 */
class StmtNode : public ASTNode {
    friend ID;

    ID id_;
    Metadata metadata_;

  public:
    void setId(const ID &id = ID::make());
    ID id() const;

    const Metadata &metadata() const { return metadata_; }
    Metadata &metadata() { return metadata_; }

    bool isStmt() const override { return true; }

    /**
     * For, If and Assert are control flow, while StmtSeq, VarDef and Assume are
     * not
     */
    virtual bool isCtrlFlow() const { return false; }

    virtual std::vector<Ref<StmtNode>> children() const { return {}; }

    /**
     * Parent, next or previous statement
     *
     * @{
     */
    Ref<StmtNode> parentStmt() const;
    Ref<StmtNode>
    parentStmtByFilter(const std::function<bool(const Stmt &)> &filter) const;
    Ref<StmtNode> prevStmt() const;
    Ref<StmtNode> nextStmt() const;
    /** @} */

    /**
     * Parent, next or previous statment, ignoring VarDef, StmtSeq or Assume
     * nodes
     *
     * @{
     */
    Ref<StmtNode> parentCtrlFlow() const;
    Ref<StmtNode> prevInCtrlFlow() const;
    Ref<StmtNode> nextInCtrlFlow() const;
    /** @} */

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

/**
 * Construct TransformedMetadata with specified operation and source Stmts.
 *
 * The children Metadatas are retrieved from the provided Stmts.
 *
 * @param op operation of the TransformedMetadata
 * @param sourceStmts variadic parameters that accept the source Stmts.
 */
template <typename... Srcs>
requires(std::convertible_to<Srcs, Stmt> &&...) auto makeMetadata(
    const std::string &op, Srcs &&...sourceStmts) {
    auto metadataFrom = [](const Stmt &s) -> Metadata {
        if (s->metadata().isValid())
            return s->metadata();
        else
            return makeMetadata(s->id());
    };
    return makeMetadata(op,
                        std::vector<Metadata>{metadataFrom(sourceStmts)...});
}

} // namespace freetensor

namespace std {

template <> struct hash<freetensor::StmtOrExprID> {
    size_t operator()(const freetensor::StmtOrExprID &id) const;
};

} // namespace std

#endif // FREE_TENSOR_AST_H
