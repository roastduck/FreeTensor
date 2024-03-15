#include <analyze/symbol_table.h>
#include <hash.h>
#include <mutator.h>
#include <numeric>
#include <pass/shrink_for.h>
#include <schedule/lower_cutlass_micro_block.h>
#include <schedule/var_merge.h>
#include <schedule/var_split.h>
#include <schedule/var_unsqueeze.h>

namespace freetensor {

namespace {

bool isPowerOfTwo(int x) { return (x & (x - 1)) == 0; }

class FixTransposeAndGetPartition : public Mutator {
    ID matMulId_;
    int64_t nWarpBatch_ = 0, nWarpM_ = 0, nWarpN_ = 0;

  public:
    FixTransposeAndGetPartition(const ID &matMulId) : matMulId_(matMulId) {}

    auto nWarpBatch() const { return nWarpBatch_; }
    auto nWarpM() const { return nWarpM_; }
    auto nWarpN() const { return nWarpN_; }

  private:
    std::tuple<int, int, int> computeWarpPartition(int64_t batch, int64_t m,
                                                   int64_t n, int64_t k,
                                                   int nWarp) {
        // Try to achieve the following goal in priority:
        //
        // 1. There should not be wasted warps, which means `nWarpM` and
        // `nWarpN` should divide `nWarp`.
        // 2. Use as more warps for the batch dimension as possible.
        // 3. `m / nWarpM` and `n / nWarpN` should be as close as possible, to
        // make the reuse in registers more efficient.

        int nWarpBatch = std::gcd(nWarp, batch);
        nWarp /= nWarpBatch;

        int nWarpM = 1, nWarpN = 1;
        if (isPowerOfTwo(nWarp)) {
            for (int i = 1; i < nWarp; i <<= 1) {
                bool mDivisible = m % 2 == 0;
                bool nDivisible = n % 2 == 0;
                if (mDivisible && nDivisible) {
                    if (m / nWarpM > n / nWarpN) {
                        nWarpM *= 2;
                    } else {
                        nWarpN *= 2;
                    }
                } else if (mDivisible) {
                    nWarpM *= 2;
                } else if (nDivisible) {
                    nWarpN *= 2;
                } else {
                    throw InvalidSchedule(
                        "Cannot compute warp partition for m = " +
                        std::to_string(m) + ", n = " + std::to_string(n) +
                        ", nWarp = " + std::to_string(nWarp));
                }
            }
        } else {
            ASSERT(false);
        }

        return {nWarpBatch, nWarpM, nWarpN};
    }

  protected:
    Stmt visit(const MatMul &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::MatMul);
        auto op = __op.as<MatMulNode>();

        if (op->id() == matMulId_) {
            ASSERT(op->backend_ == MatMulBackend::CutlassMicroBlock);

            // C is only supported for densely packed row-major layout in
            // registers
            if (!op->cIsRowMajor_) {
                op->aIsRowMajor_ = !op->aIsRowMajor_;
                op->bIsRowMajor_ = !op->bIsRowMajor_;
                op->cIsRowMajor_ = true;
                std::swap(op->aIsRowMajor_, op->bIsRowMajor_);
                std::swap(op->a_, op->b_);
                std::swap(op->lda_, op->ldb_);
                std::swap(op->stridea_, op->strideb_);
                std::swap(op->n_, op->m_);
            }

            // For a single `MatMul`, `nWarp` parameter affects the performance,
            // but the effect is limited. However, when there are multiple
            // `MatMul`s, or when there are user's threaded code in the same
            // kernel, it is critical to make `nWarp` of each of them
            // consistent, to avoid wasting warps. TODO: find a way to adjust
            // `nWarp` across different `MatMul`s.
            const int nWarp = 4; // 128 threads

            int64_t batch, m, n, k;
            if (op->batchSize_->nodeType() == ASTNodeType::IntConst) {
                batch = op->batchSize_.as<IntConstNode>()->val_;
            } else {
                throw InvalidSchedule(
                    "Dynamic size of `batchSize` is not "
                    "supported for CutlassMicroBlock backend");
            }
            if (op->m_->nodeType() == ASTNodeType::IntConst) {
                m = op->m_.as<IntConstNode>()->val_;
            } else {
                throw InvalidSchedule(
                    "Dyanmic size of `m` is not supported for "
                    "CutlassMicroBlock backend");
            }
            if (op->n_->nodeType() == ASTNodeType::IntConst) {
                n = op->n_.as<IntConstNode>()->val_;
            } else {
                throw InvalidSchedule(
                    "Dyanmic size of `n` is not supported for "
                    "CutlassMicroBlock backend");
            }
            if (op->k_->nodeType() == ASTNodeType::IntConst) {
                k = op->k_.as<IntConstNode>()->val_;
            } else {
                throw InvalidSchedule(
                    "Dyanmic size of `k` is not supported for "
                    "CutlassMicroBlock backend");
            }

            std::tie(nWarpBatch_, nWarpM_, nWarpN_) =
                computeWarpPartition(batch, m, n, k, nWarp);
        }

        return op;
    }
};

class LowerCutlassMicroBlock : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    ID matMulId_;
    int64_t nWarpBatch_ = 0, nWarpM_ = 0, nWarpN_ = 0;

    Ref<CutlassMicroKernelProperty> prop_;
    bool inMicroKernel_ = false;

  public:
    LowerCutlassMicroBlock(const ID &matMulId, int64_t nWarpBatch,
                           int64_t nWarpM, int64_t nWarpN)
        : matMulId_(matMulId), nWarpBatch_(nWarpBatch), nWarpM_(nWarpM),
          nWarpN_(nWarpN) {}

  private:
    template <typename T> Stmt guardWriteByPartition(const T &op) {
        auto ret = BaseClass::visit(op);
        if (inMicroKernel_) {
            int nDimsCAll = op->indices_.size();
            ASSERT(nDimsCAll >=
                   9); // See comments in `lowerCutlassMicroBlock` below
            auto batchInWarpPartition =
                makeEQ(op->indices_[nDimsCAll - 9], prop_->warpIdBatch_);
            auto mInWarpPartition =
                makeEQ(op->indices_[nDimsCAll - 7], prop_->warpIdM_);
            auto nInWarpPartition =
                makeEQ(op->indices_[nDimsCAll - 4], prop_->warpIdN_);
            auto mInThreadPartition =
                makeEQ(op->indices_[nDimsCAll - 5],
                       makeFloorDiv(prop_->laneId_, makeIntConst(4)));
            auto nInThreadPartition =
                makeEQ(op->indices_[nDimsCAll - 2],
                       makeMod(prop_->laneId_, makeIntConst(4)));

            ret = makeIf(
                makeLAnd(makeLAnd(batchInWarpPartition,
                                  makeLAnd(mInWarpPartition, nInWarpPartition)),
                         makeLAnd(mInThreadPartition, nInThreadPartition)),
                ret);
        }
        return ret;
    }

  protected:
    using BaseClass::visit;

    Stmt visit(const MatMul &_op) override {
        if (_op->id() == matMulId_) {
            if (inMicroKernel_) {
                throw InvalidSchedule("Micro kernels cannot nest each other");
            }

            // Here we use `threadIdx.x` for threads in a warp, and
            // `threadIdx.y` for warps, because putting everthing into a single
            // `threadIdx.x` will make the expressions to complicated to solve.
            // However, this brings a challenge when fusing parts of a program
            // with different different thread mappings. We need to come up with
            // better way to parallelize other parts of the program according to
            // the thread mapping here. (TODO)
            Expr warpId = makeVar(".matmul.threadIdx.y");
            Expr laneId = makeVar(".matmul.threadIdx.x");
            Expr warpIdBatch =
                makeFloorDiv(warpId, makeIntConst(nWarpN_ * nWarpM_));
            Expr warpIdM = makeMod(makeFloorDiv(warpId, makeIntConst(nWarpN_)),
                                   makeIntConst(nWarpM_));
            Expr warpIdN = makeMod(warpId, makeIntConst(nWarpN_));

            prop_ = Ref<CutlassMicroKernelProperty>::make(
                nWarpBatch_, nWarpM_, nWarpN_, warpIdBatch, warpIdM, warpIdN,
                laneId);

            inMicroKernel_ = true;
            auto __op = BaseClass::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::MatMul);
            auto op = __op.as<MatMulNode>();
            inMicroKernel_ = false;

            // point the c_ pointer to the starting address of each thread
            ASSERT(op->c_->nodeType() == ASTNodeType::Load);
            auto c = op->c_.as<LoadNode>();
            int nDimsCAll = c->indices_.size();
            ASSERT(nDimsCAll >=
                   9); // See comments in `lowerCutlassMicroBlock` below
            c->indices_[nDimsCAll - 9] = warpIdBatch;
            c->indices_[nDimsCAll - 7] = warpIdM;
            c->indices_[nDimsCAll - 5] = makeFloorDiv(laneId, makeIntConst(4));
            c->indices_[nDimsCAll - 4] = warpIdN;
            c->indices_[nDimsCAll - 2] = makeMod(laneId, makeIntConst(4));

            op->backend_ = MatMulBackend::CutlassMicroThread;
            op->cutlassMicroKernelProperty_ = prop_;

            auto metadata = std::move(op->metadata());
            op->metadata() = nullptr;

            const int warpSize = 32;
            Stmt ret = op;
            ret = makeFor(".matmul.threadIdx.x", makeIntConst(0),
                          makeIntConst(warpSize), makeIntConst(1),
                          makeIntConst(warpSize),
                          Ref<ForProperty>::make()->withParallel(threadIdxX),
                          std::move(ret));
            ret = makeFor(".matmul.threadIdx.y", makeIntConst(0),
                          makeIntConst(nWarpBatch_ * nWarpM_ * nWarpN_),
                          makeIntConst(1),
                          makeIntConst(nWarpBatch_ * nWarpM_ * nWarpN_),
                          Ref<ForProperty>::make()->withParallel(threadIdxY),
                          std::move(ret), std::move(metadata));
            return ret;
        } else {
            return BaseClass::visit(_op);
        }
    }

    Stmt visit(const Store &op) override { return guardWriteByPartition(op); }
    Stmt visit(const ReduceTo &op) override {
        return guardWriteByPartition(op);
    }
};

} // Anonymous namespace

Stmt lowerCutlassMicroBlock(const Stmt &_ast, const ID &matMulId,
                            const ID &defIdC,
                            const std::vector<bool> &dimsCBatch,
                            const std::vector<bool> &dimsCM,
                            const std::vector<bool> &dimsCN) {
    // Get partition info
    FixTransposeAndGetPartition fixTransposeAndGetPartition{matMulId};
    auto ast = fixTransposeAndGetPartition(_ast);
    auto nWarpBatch = fixTransposeAndGetPartition.nWarpBatch();
    auto nWarpM = fixTransposeAndGetPartition.nWarpM();
    auto nWarpN = fixTransposeAndGetPartition.nWarpN();

    // Partition C to each threads by layout-manipulation schedules. We have
    // checked we are in TryVarReorder mode in schedule/as_matmul.cc. The
    // resulting layout will be [
    //   ...: other leading dims,
    //   -9: batch warps,
    //   -8: batch serial,
    //   -7: m warps,
    //   -6: m 8-tiles,
    //   -5: m threads,
    //   -4: n warps,
    //   -3: n 8-tiles,
    //   -2: n threads,
    //   -1: n 2-tiles
    // ]
    //
    // See
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    // for 8x8 partition inside warps
    int nDimsCBatch = std::ranges::count(dimsCBatch, true);
    int nDimsCM = std::ranges::count(dimsCM, true);
    int nDimsCN = std::ranges::count(dimsCN, true);
    if (!std::all_of(dimsCN.end() - nDimsCN, dimsCN.end(),
                     [](bool b) { return b; })) {
        throw InvalidSchedule(
            FT_MSG << "Invalid C layout for cutlass_micro_block backend");
    }
    if (!std::all_of(dimsCM.end() - nDimsCN - nDimsCM, dimsCM.end() - nDimsCN,
                     [](bool b) { return b; })) {
        throw InvalidSchedule(
            FT_MSG << "Invalid C layout for cutlass_micro_block backend");
    }
    if (!std::all_of(dimsCBatch.end() - nDimsCN - nDimsCM - nDimsCBatch,
                     dimsCBatch.end() - nDimsCN - nDimsCM,
                     [](bool b) { return b; })) {
        throw InvalidSchedule(
            FT_MSG << "Invalid C layout for cutlass_micro_block backend");
    }
    int nDimsCAll = (int)dimsCBatch.size();
    int nDimsCOthers = nDimsCAll - nDimsCBatch - nDimsCM - nDimsCN;
    if (nDimsCN > 1) {
        for (int i = nDimsCN - 2; i >= 0; i--) {
            ast =
                varMerge(ast, defIdC, nDimsCOthers + nDimsCBatch + nDimsCM + i);
        }
    } else if (nDimsCN == 0) {
        ast = varUnsqueeze(ast, defIdC, nDimsCOthers + nDimsCBatch + nDimsCM);
    }
    if (nDimsCM > 1) {
        for (int i = nDimsCM - 2; i >= 0; i--) {
            ast = varMerge(ast, defIdC, nDimsCOthers + nDimsCBatch + i);
        }
    } else if (nDimsCM == 0) {
        ast = varUnsqueeze(ast, defIdC, nDimsCOthers + nDimsCBatch);
    }
    if (nDimsCBatch > 1) {
        for (int i = nDimsCBatch - 2; i >= 0; i--) {
            ast = varMerge(ast, defIdC, nDimsCOthers + i);
        }
    } else if (nDimsCBatch == 0) {
        ast = varUnsqueeze(ast, defIdC, nDimsCOthers);
    }
    // clang-format off
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 0, VarSplitMode::FixedSize, -1, nWarpBatch);
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 2, VarSplitMode::FixedSize, -1, nWarpM);
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 3, VarSplitMode::FixedSize, 8, -1);
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 5, VarSplitMode::FixedSize, -1, nWarpN);
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 6, VarSplitMode::FixedSize, 8, -1);
    ast = varSplit(
            ast, defIdC, nDimsCOthers + 7, VarSplitMode::FixedSize, 2, -1);
    // clang-format on

    // Lower to CutlassMicroThread
    LowerCutlassMicroBlock lowerCutlassMicroBlock{matMulId, nWarpBatch, nWarpM,
                                                  nWarpN};
    ast = lowerCutlassMicroBlock(ast);

    // Simplify the equivalent_ tree to help following passes
    ast = shrinkFor(ast, matMulId, true, true);

    return ast;
}

} // namespace freetensor
