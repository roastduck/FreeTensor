#ifndef FIXED_LENGTH_FEATURE_H
#define FIXED_LENGTH_FEATURE_H

#include <analyze/structural_feature.h>
#include <visitor.h>

namespace ir {

// NOTE: Exporting these constants to PyBind11 is tedious, so please sync it
// with test_fixed_length_featuer.py manually

// Features that do not need to sample
constexpr int STANDALONE_FEATURES = 0;

// Features sampled using a iteration count
constexpr int SAMPLE_GROUPS = 10;
constexpr int SAMPLE_ITERS[SAMPLE_GROUPS] = {32,   64,   128,  256,  512,
                                             1024, 2048, 4096, 8192, 16384};
constexpr int SAMPLE_FEATURES = 26;
constexpr int FEAT_SAMP_FLOAT32_OPS = 0;
constexpr int FEAT_SAMP_INT32_OPS = 1;
constexpr int FEAT_SAMP_CPU_LOAD_CNT = 2;
constexpr int FEAT_SAMP_CPU_STORE_CNT = 3;
constexpr int FEAT_SAMP_CPU_ACCESS_CNT = 4;
constexpr int FEAT_SAMP_GPU_GLOBAL_LOAD_CNT = 5;
constexpr int FEAT_SAMP_GPU_GLOBAL_STORE_CNT = 6;
constexpr int FEAT_SAMP_GPU_GLOBAL_ACCESS_CNT = 7;
constexpr int FEAT_SAMP_GPU_SHARED_LOAD_CNT = 8;
constexpr int FEAT_SAMP_GPU_SHARED_STORE_CNT = 9;
constexpr int FEAT_SAMP_GPU_SHARED_ACCESS_CNT = 10;
constexpr int FEAT_SAMP_GPU_LOCAL_LOAD_CNT = 11;
constexpr int FEAT_SAMP_GPU_LOCAL_STORE_CNT = 12;
constexpr int FEAT_SAMP_GPU_LOCAL_ACCESS_CNT = 13;
constexpr int FEAT_SAMP_CPU_LOAD_AREA = 14;
constexpr int FEAT_SAMP_CPU_STORE_AREA = 15;
constexpr int FEAT_SAMP_CPU_ACCESS_AREA = 16;
constexpr int FEAT_SAMP_GPU_GLOBAL_LOAD_AREA = 17;
constexpr int FEAT_SAMP_GPU_GLOBAL_STORE_AREA = 18;
constexpr int FEAT_SAMP_GPU_GLOBAL_ACCESS_AREA = 19;
constexpr int FEAT_SAMP_GPU_SHARED_LOAD_AREA = 20;
constexpr int FEAT_SAMP_GPU_SHARED_STORE_AREA = 21;
constexpr int FEAT_SAMP_GPU_SHARED_ACCESS_AREA = 22;
constexpr int FEAT_SAMP_GPU_LOCAL_LOAD_AREA = 23;
constexpr int FEAT_SAMP_GPU_LOCAL_STORE_AREA = 24;
constexpr int FEAT_SAMP_GPU_LOCAL_ACCESS_AREA = 25;

/**
 * Convert a structural feature into a fixed length one, so can be consumed by
 * XGBoost
 */
class FixedLengthFeature : public Visitor {
    const std::unordered_map<std::string, NodeFeature> &structural_;
    std::unordered_map<Stmt, int64_t> iterCnts_; // -1 means unknown
    std::unordered_map<Stmt, std::vector<std::vector<double>>>
        samples_; // -1 means unknown. Passed down through the AST

  public:
    FixedLengthFeature(
        const std::unordered_map<std::string, NodeFeature> &structural)
        : structural_(structural) {}

    std::vector<double> features(const Stmt &root) const;
    static size_t feature_len();

  protected:
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
    void visit(const VarDef &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Eval &op) override;
};

inline std::vector<double> fixedLengthFeature(const Stmt &op) {
    auto structural = structuralFeature(op);
    FixedLengthFeature visitor(structural);
    visitor(op);
    return visitor.features(op);
}

} // namespace ir

#endif // FIXED_LENGTH_FEATURE_H
