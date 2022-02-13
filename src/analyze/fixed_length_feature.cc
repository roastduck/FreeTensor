#include <itertools.hpp>

#include <analyze/fixed_length_feature.h>

namespace ir {

template <class K, class V>
static V getOrDefault(const std::unordered_map<K, V> &map, const K &key,
                      const V &defaultValue = 0) {
    return map.count(key) ? map.at(key) : defaultValue;
}

static std::vector<double> interpolate(const std::vector<double> &bodyFeat,
                                       int64_t bodyIter,
                                       const std::vector<double> &totFeat,
                                       int64_t totIter, int64_t reqIter) {
    ASSERT(bodyFeat.size() == totFeat.size());
    if (reqIter >= totIter) {
        return totFeat;
    } else {
        size_t n = bodyFeat.size();
        std::vector<double> ret(n);
        for (auto &&[r, body, tot] : iter::zip(ret, bodyFeat, totFeat)) {
            if (body == -1 || tot == -1) {
                r = -1;
            } else {
                auto perReqIter = body + (tot - body) * (reqIter - bodyIter) /
                                             (totIter - bodyIter);
                r = perReqIter / reqIter * totIter;
            }
        }
        return ret;
    }
}

static std::vector<double> scale(const std::vector<double> &feat, int64_t k) {
    std::vector<double> ret;
    ret.reserve(feat.size());
    for (double val : feat) {
        ret.emplace_back(val == -1 || k == -1 ? -1 : val * k);
    }
    return ret;
}

static void mixTo(std::vector<double> &parent,
                  const std::vector<double> &child) {
    ASSERT(parent.size() == child.size());
    for (auto &&[ch, par] : iter::zip(child, parent)) {
        if (ch == -1 || par == -1) {
            par = -1;
        } else {
            par += ch;
        }
    }
}

static std::vector<double> asVec(const NodeFeature &nodeFeat) {
    std::vector<double> ret(SAMPLE_FEATURES);
    ret[FEAT_SAMP_FLOAT32_OPS] =
        getOrDefault(nodeFeat.opCnt_, DataType::Float32);
    ret[FEAT_SAMP_INT32_OPS] = getOrDefault(nodeFeat.opCnt_, DataType::Int32);
    ret[FEAT_SAMP_CPU_LOAD_CNT] = getOrDefault(nodeFeat.loadCnt_, MemType::CPU);
    ret[FEAT_SAMP_CPU_STORE_CNT] =
        getOrDefault(nodeFeat.storeCnt_, MemType::CPU);
    ret[FEAT_SAMP_CPU_ACCESS_CNT] =
        getOrDefault(nodeFeat.accessCnt_, MemType::CPU);
    ret[FEAT_SAMP_GPU_GLOBAL_LOAD_CNT] =
        getOrDefault(nodeFeat.loadCnt_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_GLOBAL_STORE_CNT] =
        getOrDefault(nodeFeat.storeCnt_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_GLOBAL_ACCESS_CNT] =
        getOrDefault(nodeFeat.accessCnt_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_SHARED_LOAD_CNT] =
        getOrDefault(nodeFeat.loadCnt_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_SHARED_STORE_CNT] =
        getOrDefault(nodeFeat.storeCnt_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_SHARED_ACCESS_CNT] =
        getOrDefault(nodeFeat.accessCnt_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_LOCAL_LOAD_CNT] =
        getOrDefault(nodeFeat.loadCnt_, MemType::GPULocal);
    ret[FEAT_SAMP_GPU_LOCAL_STORE_CNT] =
        getOrDefault(nodeFeat.storeCnt_, MemType::GPULocal);
    ret[FEAT_SAMP_GPU_LOCAL_ACCESS_CNT] =
        getOrDefault(nodeFeat.accessCnt_, MemType::GPULocal);
    ret[FEAT_SAMP_CPU_LOAD_AREA] =
        getOrDefault(nodeFeat.loadArea_, MemType::CPU);
    ret[FEAT_SAMP_CPU_STORE_AREA] =
        getOrDefault(nodeFeat.storeArea_, MemType::CPU);
    ret[FEAT_SAMP_CPU_ACCESS_AREA] =
        getOrDefault(nodeFeat.accessArea_, MemType::CPU);
    ret[FEAT_SAMP_GPU_GLOBAL_LOAD_AREA] =
        getOrDefault(nodeFeat.loadArea_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_GLOBAL_STORE_AREA] =
        getOrDefault(nodeFeat.storeArea_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_GLOBAL_ACCESS_AREA] =
        getOrDefault(nodeFeat.accessArea_, MemType::GPUGlobal);
    ret[FEAT_SAMP_GPU_SHARED_LOAD_AREA] =
        getOrDefault(nodeFeat.loadArea_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_SHARED_STORE_AREA] =
        getOrDefault(nodeFeat.storeArea_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_SHARED_ACCESS_AREA] =
        getOrDefault(nodeFeat.accessArea_, MemType::GPUShared);
    ret[FEAT_SAMP_GPU_LOCAL_LOAD_AREA] =
        getOrDefault(nodeFeat.loadArea_, MemType::GPULocal);
    ret[FEAT_SAMP_GPU_LOCAL_STORE_AREA] =
        getOrDefault(nodeFeat.storeArea_, MemType::GPULocal);
    ret[FEAT_SAMP_GPU_LOCAL_ACCESS_AREA] =
        getOrDefault(nodeFeat.accessArea_, MemType::GPULocal);
    return ret;
}

std::vector<double> FixedLengthFeature::features(const Stmt &root) const {
    std::vector<double> ret(STANDALONE_FEATURES +
                            SAMPLE_GROUPS * SAMPLE_FEATURES);
    const auto &samples = samples_.at(root);
    auto iterCnt = iterCnts_.at(root);
    for (int i = 0; i < SAMPLE_GROUPS; i++) {
        for (int j = 0; j < SAMPLE_FEATURES; j++) {
            ret[STANDALONE_FEATURES + i * SAMPLE_FEATURES + j] =
                samples[i][j] / iterCnt * SAMPLE_ITERS[i];
        }
    }
    return ret;
}

void FixedLengthFeature::visit(const StmtSeq &op) {
    Visitor::visit(op);
    int64_t totIter = 0;
    auto totStruct = asVec(structural_.at(op->id()));
    std::vector<std::vector<double>> totSample(SAMPLE_GROUPS);
    for (auto &&stmt : op->stmts_) {
        if (iterCnts_.at(stmt) == -1) {
            totIter = -1;
        } else {
            totIter += iterCnts_.at(stmt);
        }
    }
    for (int i = 0; i < SAMPLE_GROUPS; i++) {
        bool useMix = false;
        for (auto &&stmt : op->stmts_) {
            if (SAMPLE_ITERS[i] < iterCnts_.at(stmt)) {
                useMix = true;
                break;
            }
        }
        if (useMix) {
            std::vector<double> result(SAMPLE_FEATURES, 0);
            for (auto &&stmt : op->stmts_) {
                mixTo(result, samples_.at(stmt)[i]);
            }
            totSample[i] = std::move(result);
        } else {
            totSample[i] = totStruct;
        }
    }
    iterCnts_[op] = totIter;
    samples_[op] = std::move(totSample);
}

void FixedLengthFeature::visit(const For &op) {
    Visitor::visit(op);
    int64_t bodyIter = iterCnts_.at(op->body_);
    int64_t thisIter = op->len_->nodeType() == ASTNodeType::IntConst
                           ? op->len_.as<IntConstNode>()->val_
                           : -1;
    int64_t totIter =
        bodyIter == -1 || thisIter == -1 ? -1 : bodyIter * thisIter;
    auto bodyStruct = asVec(structural_.at(op->body_->id()));
    auto totStruct = asVec(structural_.at(op->id()));
    const std::vector<std::vector<double>> &bodySample = samples_.at(op->body_);
    std::vector<std::vector<double>> totSample(SAMPLE_GROUPS);
    for (int i = 0; i < SAMPLE_GROUPS; i++) {
        if (SAMPLE_ITERS[i] < bodyIter) {
            totSample[i] = scale(bodySample[i], thisIter);
        } else {
            totSample[i] = interpolate(bodyStruct, bodyIter, totStruct, totIter,
                                       SAMPLE_ITERS[i]);
        }
    }
    iterCnts_[op] = totIter;
    samples_[op] = std::move(totSample);
}

void FixedLengthFeature::visit(const If &op) {
    Visitor::visit(op);
    int64_t thenIter = iterCnts_.at(op->thenCase_);
    int64_t elseIter =
        op->elseCase_.isValid() ? iterCnts_.at(op->elseCase_) : 0;
    int64_t totIter =
        thenIter == -1 || elseIter == -1 ? -1 : thenIter + elseIter;
    auto totStruct = asVec(structural_.at(op->id()));
    std::vector<std::vector<double>> totSample(SAMPLE_GROUPS);
    for (int i = 0; i < SAMPLE_GROUPS; i++) {
        if (SAMPLE_ITERS[i] < iterCnts_.at(op->thenCase_) ||
            (op->elseCase_.isValid() &&
             SAMPLE_ITERS[i] < iterCnts_.at(op->elseCase_))) {
            std::vector<double> result(SAMPLE_FEATURES, 0);
            mixTo(result, samples_.at(op->thenCase_)[i]);
            if (op->elseCase_.isValid()) {
                mixTo(result, samples_.at(op->elseCase_)[i]);
            }
            totSample[i] = std::move(result);
        } else {
            totSample[i] = totStruct;
        }
    }
    iterCnts_[op] = totIter;
    samples_[op] = std::move(totSample);
}

void FixedLengthFeature::visit(const Assert &op) {
    Visitor::visit(op);
    iterCnts_[op] = iterCnts_.at(op->body_);
    samples_[op] = samples_.at(op->body_);
}

void FixedLengthFeature::visit(const VarDef &op) {
    Visitor::visit(op);
    iterCnts_[op] = iterCnts_.at(op->body_);
    samples_[op] = samples_.at(op->body_);
}

void FixedLengthFeature::visit(const Store &op) {
    Visitor::visit(op);
    iterCnts_[op] = 1;
    samples_[op].resize(SAMPLE_GROUPS, asVec(structural_.at(op->id())));
}

void FixedLengthFeature::visit(const ReduceTo &op) {
    Visitor::visit(op);
    iterCnts_[op] = 1;
    samples_[op].resize(SAMPLE_GROUPS, asVec(structural_.at(op->id())));
}

void FixedLengthFeature::visit(const Eval &op) {
    Visitor::visit(op);
    iterCnts_[op] = 1;
    samples_[op].resize(SAMPLE_GROUPS, asVec(structural_.at(op->id())));
}

size_t FixedLengthFeature::featureLen() {
    return STANDALONE_FEATURES + SAMPLE_GROUPS * SAMPLE_FEATURES;
}

} // namespace ir
