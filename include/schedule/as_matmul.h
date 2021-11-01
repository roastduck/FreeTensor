#ifndef MAKE_MATMUL_H
#define MAKE_MATMUL_H

#include <unordered_map>
#include <unordered_set>

#include <analyze/analyze_linear.h>
#include <analyze/check_all_defined.h>
#include <analyze/hash.h>
#include <mutator.h>

namespace ir {

class AsMatMul : public Mutator {
    std::string loop_;

    int nestCnt_ = 0;
    std::vector<For> nests_;
    std::unordered_map<std::string, int> iterMap_; // iter var -> nest cnt
    std::unordered_map<std::string, Ref<Buffer>> buffers_; // var name -> buffer
    std::unordered_set<std::string> outerDefs_;
    std::vector<int> orderInit_;

    bool foundInit_ = false, foundLeaf_ = false, inside_ = false;
    Expr a_, b_, c_, initC_, m_, k_, n_, lda_, stridea_, ldb_, strideb_, ldc_,
        stridec_, batchSize_;
    bool aIsRowMajor_, bIsRowMajor_, cIsRowMajor_;

    GetHash getHash_;
    AnalyzeLinear analyzeLinear_;

  public:
    AsMatMul(const std::string &loop) : loop_(loop) {}

  private:
    uint64_t getHash(const Expr &op);
    const LinearExpr<int> &analyzeLinear(const Expr &expr);

    template <class T>
    std::tuple<std::vector<bool>, std::vector<int>, Expr>
    findIterUsedAndBaseAddr(const T &acc) {
        std::vector<bool> usedBy(nestCnt_, false);
        std::vector<int> order;
        Expr baseAddr = makeLoad(acc->var_, acc->indices_);
        for (size_t i = 0, n = acc->indices_.size(); i < n; i++) {
            auto &&idx = acc->indices_[i];
            auto &&lin = analyzeLinear(idx);
            if (lin.coeff_.size() != 1 ||
                std::abs(lin.coeff_.front().second.k_) != 1 ||
                lin.coeff_.front().second.a_->nodeType() != ASTNodeType::Var) {
                if (!checkAllDefined(outerDefs_, idx)) {
                    throw InvalidSchedule("Indices of " + acc->var_ +
                                          " should be plain loop iterators");
                }
                continue; // not a dim in matmul
            }
            Var var = lin.coeff_.front().second.a_.template as<VarNode>();
            if (!iterMap_.count(var->name_)) {
                continue; // not a dim in matmul
            } else {
                baseAddr.as<LoadNode>()->indices_[i] = makeIntConst(0);
            }
            int loop = iterMap_.at(var->name_);
            if (getHash(nests_[loop]->len_) !=
                getHash(buffers_.at(acc->var_)->tensor().shape()[i])) {
                throw InvalidSchedule(
                    "Iterator " + var->name_ + " of " + acc->var_ +
                    " should loop over the entire range (" +
                    toString(buffers_.at(acc->var_)->tensor().shape()[i]) +
                    "), instead of " + toString(nests_[loop]->len_));
            }
            usedBy[loop] = true;
            order.emplace_back(loop);
        }
        return std::make_tuple(usedBy, order, baseAddr);
    }

    template <class T>
    std::pair<Expr, Expr> findLenAndStride(const T &acc,
                                           const std::vector<bool> &flag) {
        Expr len, stride;
        bool thisDimIn = false, lastDimIn = false;
        Expr lastInDim;
        for (size_t i = 0, n = acc->indices_.size(); i < n; i++) {
            auto &&idx = acc->indices_[i];
            auto &&lin = analyzeLinear(idx);
            lastDimIn = thisDimIn;
            thisDimIn = true;
            if (lin.coeff_.size() != 1 ||
                std::abs(lin.coeff_.front().second.k_) != 1 ||
                lin.coeff_.front().second.a_->nodeType() != ASTNodeType::Var) {
                thisDimIn = false;
            } else {
                Var var = lin.coeff_.front().second.a_.template as<VarNode>();
                if (!iterMap_.count(var->name_) ||
                    !flag[iterMap_.at(var->name_)]) {
                    thisDimIn = false;
                }
            }
            if (thisDimIn) {
                if (lastInDim.isValid()) {
                    if (!lastDimIn) {
                        throw InvalidSchedule(
                            "Dimensions " + toString(lastInDim) + " and " +
                            toString(idx) + " should be contiguous");
                    }
                }
                auto thisLen = buffers_.at(acc->var_)->tensor().shape()[i];
                len = len.isValid() ? makeMul(len, thisLen) : (Expr)thisLen;
                lastInDim = idx;
            } else {
                if (len.isValid()) {
                    auto thisLen = buffers_.at(acc->var_)->tensor().shape()[i];
                    stride = stride.isValid() ? makeMul(stride, thisLen)
                                              : (Expr)thisLen;
                }
            }
        }
        len = len.isValid() ? len : makeIntConst(1);
        stride = stride.isValid() ? stride : makeIntConst(1);
        return std::make_pair(len, stride);
    }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const For &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt asMatMul(const Stmt &ast, const std::string &loop);

} // namespace ir

#endif // MAKE_MATMUL_H
