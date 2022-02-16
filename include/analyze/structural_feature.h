#ifndef STRUCTURAL_FEATURE_H
#define STRUCTURAL_FEATURE_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/comp_transient_bounds.h>
#include <analyze/comp_unique_bounds.h>
#include <analyze/symbol_table.h>
#include <analyze/type_infer.h>
#include <visitor.h>

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
class StructuralFeature
    : public CompTransientBounds<WithTypeInfer<SymbolTable<Visitor>>> {
    typedef CompTransientBounds<WithTypeInfer<SymbolTable<Visitor>>> BaseClass;

    /**
     * Memory access info of an AST node with respect to a buffer
     */
    struct NodeBufferInfo {
        std::vector<CompUniqueBounds::LowerBoundsList> lo_;
        std::vector<CompUniqueBounds::UpperBoundsList> hi_;
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

    CompUniqueBounds bound_;

    std::unordered_map<ID, NodeFeature> features_; // Node ID -> features
    std::unordered_map<AST, NodeInfo> info_;       // AST -> info

  public:
    StructuralFeature() : bound_(*this, *this) {}

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

} // namespace ir

#endif // STRUCTURAL_FEATURE_H
