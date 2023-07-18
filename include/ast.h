#ifndef FREE_TENSOR_AST_H
#define FREE_TENSOR_AST_H

#include <atomic>
#include <functional>
#include <iostream>
#include <optional>
#include <source_location>
#include <string>

#include <id.h>
#include <metadata.h>
#include <ref.h>
#include <serialize/to_string.h>
#include <sub_tree.h>
#include <type/data_type.h>

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
    Ln,
    Square,
    Sigmoid,
    Sin,
    Cos,
    Tan,
    Tanh,
    Abs,
    Floor,
    Ceil,
    Unbound,

    // Other expressions
    IfExpr,
    Cast,
    Intrinsic,

    // For custom gradient only
    MarkVersion,
    LoadAtVersion,
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
#ifdef FT_DEBUG_BLAME_AST
    std::source_location debugBlame_;
#endif

  public:
    virtual ~ASTNode() {}
    virtual ASTNodeType nodeType() const = 0;

    bool isAST() const override { return true; }
    virtual bool isFunc() const { return false; }
    virtual bool isStmt() const { return false; }
    virtual bool isExpr() const { return false; }

    Ref<ASTNode> parentAST() const;

    std::source_location debugBlame() const {
#ifdef FT_DEBUG_BLAME_AST
        return debugBlame_;
#else
        return std::source_location::current(); // Arbitrary return
#endif
    }
    void setDebugBlame(std::source_location loc) {
#ifdef FT_DEBUG_BLAME_AST
        debugBlame_ = loc;
#endif
    }

    DEFINE_NODE_ACCESS(AST);
};
typedef Ref<ASTNode> AST;

class StmtNode;
typedef Ref<StmtNode> Stmt;

/**
 * Base class of all expression nodes in an AST
 */
class ExprNode : public ASTNode {
  protected:
    std::optional<DataType> dtype_;

  public:
    bool isExpr() const override { return true; }

    virtual bool isConst() const { return false; }
    virtual bool isBinary() const { return false; }
    virtual bool isUnary() const { return false; }

    virtual std::vector<Ref<ExprNode>> children() const = 0;

    Ref<ExprNode> parentExpr() const;
    Ref<StmtNode> parentStmt() const;

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
    StmtOrExprID() {}

    StmtOrExprID(const ID &stmtId) : stmtId_(stmtId) {}

    StmtOrExprID(const Expr &expr, const ID &stmtId)
        : stmtId_(stmtId), expr_(expr) {}

    template <std::convertible_to<Stmt> T>
    StmtOrExprID(const Expr &expr, T &&parent) : stmtId_(parent->id()) {
        expr_ = expr;
    }

    const ID &stmtId() const { return stmtId_; }
    const Expr &expr() const { return expr_; }

    bool isValid() const { return stmtId_.isValid(); }

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
     * NOTE: For an If node, the "then" case is considered before the "else"
     * case
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
     * Previous or next statement in DFS order
     */
    Ref<StmtNode> prevLeafStmtInDFSOrder() const;
    Ref<StmtNode> nextLeafStmtInDFSOrder() const;
    Ref<StmtNode> prevStmtInDFSPostOrder() const; // may return child
    Ref<StmtNode> nextStmtInDFSPreOrder() const;  // may return child

    /**
     * Find an ancestor by ID. `this` itself is also considered
     */
    Ref<StmtNode> ancestorById(const ID &lookup) const;

    /**
     * Check whether this node is an ancestoer of `other`
     */
    bool isAncestorOf(const Stmt &other) const;

    /**
     * Check whether this node is before `other` in DFS order
     */
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
    requires(std::convertible_to<Srcs, Stmt> && ...)
auto makeMetadata(const std::string &op, Srcs &&...sourceStmts) {
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
