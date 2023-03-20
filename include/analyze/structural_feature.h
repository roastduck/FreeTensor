#ifndef FREE_TENSOR_STRUCTURAL_FEATURE_H
#define FREE_TENSOR_STRUCTURAL_FEATURE_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_access_bound.h>
#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <visitor.h>

namespace freetensor {

/**
 * Features of a statement node
 */
struct NodeFeature {
    // -1 means unknown
    std::unordered_map<BaseDataType, int64_t> opCnt_;
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
 *
 * This Visitor generate a NodeFeature for each node, depicting the performance
 * feature of running the subtree rooted at the node
 */
class StructuralFeature : public CompTransientBounds<SymbolTable<Visitor>> {
    typedef CompTransientBounds<SymbolTable<Visitor>> BaseClass;

    /**
     * Info about an AST node, but not necessarily a feature
     */
    struct NodeInfo {
        std::unordered_map<BaseDataType, int64_t> opCnt_;
        std::unordered_map<MemType, int64_t> loadCnt_, storeCnt_, accessCnt_;

        std::unordered_map<MemType, int64_t> innerLoadArea_, innerStoreArea_,
            innerAccessArea_;

        // buffer name -> all accesses of this buffer
        // If a key does not exist in loads_, stores_ or accesses_, it
        // means that a node does not access the buffer
        std::unordered_map<std::string, std::vector<CompAccessBound::Access>>
            loads_, stores_, accesses_;
    };

    CompUniqueBounds bound_;

    std::unordered_map<ID, NodeFeature> features_; // Node ID -> features
    std::unordered_map<AST, NodeInfo> info_;       // AST -> info

  public:
    StructuralFeature() : bound_(*this) {}

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
    int64_t calcArea(const std::string &var,
                     const std::vector<CompAccessBound::Access> &accesses);
    void calcAreaFeatures(const Stmt &parent);
    void calcFeatures(const Stmt &parent);

    void visitBinOp(const BinaryExpr &op);
    void visitUnaryOp(const UnaryExpr &op);

  protected:
    using BaseClass::visit;

    void visitStmt(const Stmt &op) override;
    void visitExpr(const Expr &op) override;

    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;

    void visit(const Cast &op) override;
    void visit(const IfExpr &op) override;

    void visit(const StmtSeq &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
    void visit(const For &op) override;
    void visit(const VarDef &op) override;
};

inline std::unordered_map<ID, NodeFeature> structuralFeature(const Stmt &op) {
    StructuralFeature visitor;
    visitor(op);
    return visitor.features();
}

} // namespace freetensor

#endif // FREE_TENSOR_STRUCTURAL_FEATURE_H
