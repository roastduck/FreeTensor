#ifndef MAKE_MATMUL_H
#define MAKE_MATMUL_H

#include <unordered_map>

#include <analyze/hash.h>
#include <mutator.h>

namespace ir {

class AsMatMul : public Mutator {
    std::string loop_;

    int nestCnt_ = 0;
    std::vector<For> nests_;
    std::unordered_map<std::string, int> iterMap_; // iter var -> nest cnt
    std::unordered_map<std::string, Ref<Buffer>> buffers_; // var name -> buffer

    bool foundInit_ = false, foundLeaf_ = false, inside_ = false;
    Expr a_, b_, c_, initC_, m_, k_, n_, lda_, stridea_, ldb_, strideb_, ldc_,
        stridec_, batchSize_;
    bool aIsRowMajor_, bIsRowMajor_, cIsRowMajor_;

    GetHash getHash_;

  public:
    AsMatMul(const std::string &loop) : loop_(loop) {}

  private:
    uint64_t getHash(const Expr &op);

    template <class T>
    std::pair<std::vector<bool>, Expr> findIterUsedAndBaseAddr(const T &acc) {
        std::vector<bool> usedBy(nestCnt_, false);
        Expr baseAddr = makeLoad(acc->var_, acc->indices_);
        for (size_t i = 0, n = acc->indices_.size(); i < n; i++) {
            auto &&idx = acc->indices_[i];
            if (idx->nodeType() != ASTNodeType::Var) {
                throw InvalidSchedule("Indices of " + acc->var_ +
                                      " should be plain loop iterators");
            }
            Var var = idx.template as<VarNode>();
            if (!iterMap_.count(var->name_)) {
                continue;
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
        }
        return std::make_pair(usedBy, baseAddr);
    }

    template <class T>
    std::pair<Expr, Expr> findLenAndStride(const T &acc,
                                           const std::vector<bool> &flag) {
        Expr len, stride;
        for (size_t i = 0, n = acc->indices_.size(); i < n; i++) {
            ASSERT(acc->indices_[i]->nodeType() == ASTNodeType::Var);
            Var var = acc->indices_[i].template as<VarNode>();
            if (iterMap_.count(var->name_) && flag[iterMap_.at(var->name_)]) {
                if (len.isValid()) { // started
                    ASSERT(acc->indices_[i - 1]->nodeType() ==
                           ASTNodeType::Var);
                    Var lastVar = acc->indices_[i - 1].template as<VarNode>();
                    int lastLoop = iterMap_.at(lastVar->name_);
                    if (!flag[lastLoop]) {
                        throw InvalidSchedule("Dimensions " + lastVar->name_ +
                                              " and " + var->name_ +
                                              " should be contiguous");
                    }
                }
                auto thisLen = buffers_.at(acc->var_)->tensor().shape()[i];
                len = len.isValid() ? makeMul(len, thisLen) : (Expr)thisLen;
            } else if (len.isValid()) {
                auto thisLen = buffers_.at(acc->var_)->tensor().shape()[i];
                stride =
                    stride.isValid() ? makeMul(stride, thisLen) : (Expr)thisLen;
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

} // namespace ir

#endif // MAKE_MATMUL_H
