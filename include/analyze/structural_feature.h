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

    std::unordered_map<ID, NodeFeature> features_; // Node ID -> features
    std::unordered_map<AST, NodeInfo> info_;       // AST -> info

  public:
    const std::unordered_map<ID, NodeFeature> &features() const {
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

    Expr visitBinOp(const BinaryExpr &_op);
    Expr visitUnaryOp(const UnaryExpr &_op);

  protected:
    using CompUniqueBounds::visit;

    Stmt visitStmt(const Stmt &op) override;
    Expr visitExpr(const Expr &op) override;

    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;

    Expr visit(const Cast &op) override;
    Expr visit(const IfExpr &op) override;

    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

inline std::unordered_map<ID, NodeFeature> structuralFeature(const Stmt &op) {
    StructuralFeature visitor; // actually a Mutator, but we drop the result
    visitor(op);
    return visitor.features();
}

} // namespace ir

#endif // STRUCTURAL_FEATURE_H
