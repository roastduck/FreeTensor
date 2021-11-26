#ifndef STRUCTURAL_FEATURE_H
#define STRUCTURAL_FEATURE_H

#include <unordered_map>
#include <unordered_set>

#include <pass/simplify.h>

namespace ir {

/**
 * Features of a statement node
 */
struct NodeFeature {
    // -1 means unknown
    std::unordered_map<DataType, int64_t> opCnt_;
    std::unordered_map<MemType, int64_t> loadCnt_, storeCnt_,
        accessCnt_; // Memory access count
    std::unordered_map<MemType, int64_t> loadArea_, storeArea_,
        accessArea_; // memory footprint
};

/**
 * Analyze program features for each nodes
 *
 * Program features are used by machine learning models to predict program
 * performance. This pass outputs an structual feature, which can be converted
 * to a plain feature by a following pass
 */
class StructuralFeature : public CompUniqueBounds {
    typedef CompUniqueBounds
        BaseClass; // Replace it with any simplifying pass if needed

    /**
     * Memory access info of an AST node with respect to a buffer
     */
    struct NodeBufferInfo {
        std::vector<LowerBoundsList> lo_;
        std::vector<UpperBoundsList> hi_;
    };

    /**
     * Info about an AST node, but not necessarily a feature
     */
    struct NodeInfo {
        std::unordered_map<DataType, int64_t> opCnt_;
        std::unordered_map<MemType, int64_t> loadCnt_, storeCnt_, accessCnt_;

        std::unordered_map<MemType, int64_t> innerLoadArea_, innerStoreArea_,
            innerAccessArea_;
        std::unordered_map<std::string, NodeBufferInfo> loads_, stores_,
            accesses_; // buffer name -> buffer info
        // NOTE: If a key does not exist in loads_, stores_ or accesses_, it
        // means that a node does not access the buffer. But, if there is an
        // empty NodeBufferInfo, it means there is an access whose indices is
        // unlimited
    };

    std::unordered_map<std::string, NodeFeature>
        features_;                           // Node ID -> features
    std::unordered_map<AST, NodeInfo> info_; // AST -> info

    std::unordered_set<std::string>
        defs_; // All names currently defined in a subtree
    std::unordered_map<std::string, Ref<Buffer>>
        buffers_; // All buffers currently defined in a subtree

  public:
    const std::unordered_map<std::string, NodeFeature> &features() const {
        return features_;
    }

  private:
    void updCompInfo(const AST &parent, const AST &child, int repeat = 1);
    void updAccCntInfo(const AST &parent, const AST &child, int repeat = 1);
    void updAreaInfo(const AST &parent, const AST &child);
    void updInfo(const AST &parent, const AST &child,
                 int repeat = 1); // repeat = -1 means unknown

    void calcCompFeatures(const Stmt &parent);
    void calcAccCntFeatures(const Stmt &parent);
    int64_t calcArea(const NodeBufferInfo &bufInfo);
    void calcAreaFeatures(const Stmt &parent);
    void calcFeatures(const Stmt &parent);

    template <class T> Expr visitBinOp(const T &_op) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        updInfo(op, op->lhs_);
        updInfo(op, op->rhs_);
        info_[op].opCnt_[upCast(dtype(op->lhs_), dtype(op->rhs_))]++;
        return op;
    }

    template <class T> Expr visitUnaryOp(const T &_op) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        updInfo(op, op->expr_);
        info_[op].opCnt_[dtype(op->expr_)]++;
        return op;
    }

  protected:
    using CompUniqueBounds::visit;

    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;

    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;

    Expr visit(const Add &op) override { return visitBinOp(op); }
    Expr visit(const Sub &op) override { return visitBinOp(op); }
    Expr visit(const Mul &op) override { return visitBinOp(op); }
    Expr visit(const RealDiv &op) override { return visitBinOp(op); }
    Expr visit(const FloorDiv &op) override { return visitBinOp(op); }
    Expr visit(const CeilDiv &op) override { return visitBinOp(op); }
    Expr visit(const RoundTowards0Div &op) override { return visitBinOp(op); }
    Expr visit(const Mod &op) override { return visitBinOp(op); }
    Expr visit(const Remainder &op) override { return visitBinOp(op); }
    Expr visit(const Min &op) override { return visitBinOp(op); }
    Expr visit(const Max &op) override { return visitBinOp(op); }
    Expr visit(const LT &op) override { return visitBinOp(op); }
    Expr visit(const LE &op) override { return visitBinOp(op); }
    Expr visit(const GT &op) override { return visitBinOp(op); }
    Expr visit(const GE &op) override { return visitBinOp(op); }
    Expr visit(const EQ &op) override { return visitBinOp(op); }
    Expr visit(const NE &op) override { return visitBinOp(op); }
    Expr visit(const LAnd &op) override { return visitBinOp(op); }
    Expr visit(const LOr &op) override { return visitBinOp(op); }
    Expr visit(const LNot &op) override { return visitUnaryOp(op); }
    Expr visit(const Sqrt &op) override { return visitUnaryOp(op); }
    Expr visit(const Exp &op) override { return visitUnaryOp(op); }
    Expr visit(const Square &op) override { return visitUnaryOp(op); }
    Expr visit(const Sigmoid &op) override { return visitUnaryOp(op); }
    Expr visit(const Tanh &op) override { return visitUnaryOp(op); }
    Expr visit(const Abs &op) override { return visitUnaryOp(op); }
    Expr visit(const Floor &op) override { return visitUnaryOp(op); }
    Expr visit(const Ceil &op) override { return visitUnaryOp(op); }
    Expr visit(const Cast &op) override { return visitUnaryOp(op); }
    Expr visit(const IfExpr &op) override;

    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

inline std::unordered_map<std::string, NodeFeature>
structuralFeature(const Stmt &op) {
    StructuralFeature visitor; // actually a Mutator, but we drop the result
    visitor(op);
    return visitor.features();
}

} // namespace ir

#endif // STRUCTURAL_FEATURE_H
