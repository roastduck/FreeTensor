#include <algorithm>
#include <optional>

#include <config.h>
#include <pass/simplify.h>
#include <schedule.h>
#include <schedule/as_matmul.h>

namespace freetensor {

static bool isIntConst1(const Expr &op) {
    return op->nodeType() == ASTNodeType::IntConst &&
           op.as<IntConstNode>()->val_ == 1;
}

static bool isConst0(const Expr &op) {
    return (op->nodeType() == ASTNodeType::IntConst &&
            op.as<IntConstNode>()->val_ == 0) ||
           (op->nodeType() == ASTNodeType::FloatConst &&
            op.as<FloatConstNode>()->val_ == 0);
}

template <class T>
static std::optional<std::vector<int>> permutationB2A(const std::vector<T> &A,
                                                      const std::vector<T> &B) {
    ASSERT(A.size() == B.size());
    std::vector<int> permutation(A.size(), 0);
    auto idxAndA = ranges::to<std::vector>(views::enumerate(A));
    auto idxAndB = ranges::to<std::vector>(views::enumerate(B));
    std::ranges::stable_sort(idxAndA, [](auto &&lhs, auto &&rhs) {
        return std::get<1>(lhs) < std::get<1>(rhs);
    });
    std::ranges::stable_sort(idxAndB, [](auto &&lhs, auto &&rhs) {
        return std::get<1>(lhs) < std::get<1>(rhs);
    });
    for (auto &&[itemA, itemB] : views::zip(idxAndA, idxAndB)) {
        auto &&[idxA, valA] = itemA;
        auto &&[idxB, valB] = itemB;
        if (valA != valB) {
            return std::nullopt;
        }
        permutation[idxB] = idxA;
    }
    return permutation;
}

static std::vector<int> filter(const std::vector<int> &order,
                               const std::vector<bool> &flag) {
    std::vector<int> ret;
    ret.reserve(order.size());
    for (int item : order) {
        if (flag.at(item)) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

static std::pair<std::vector<int>, std::vector<int>>
filterAndGetIdx(const std::vector<int> &order, const std::vector<bool> &flag) {
    std::vector<int> sub, idx;
    sub.reserve(order.size());
    idx.reserve(order.size());
    for (auto &&[i, item] : views::enumerate(order)) {
        if (flag.at(item)) {
            sub.emplace_back(item);
            idx.emplace_back(i);
        }
    }
    return {sub, idx};
}

static bool inOrder(const std::vector<bool> &before,
                    const std::vector<bool> &after) {
    bool metAfter = false;
    for (auto &&[isBefore, isAfter] : views::zip(before, after)) {
        if (isAfter) {
            metAfter = true;
        }
        if (isBefore && metAfter) {
            return false;
        }
    }
    return true;
}

void AsMatMul::checkSameOrderOrRetry(const ID &idA,
                                     const std::vector<int> &orderA,
                                     const std::vector<bool> &filterA,
                                     const ID &idB,
                                     const std::vector<int> &orderB,
                                     const std::vector<bool> &filterB,
                                     const std::string &message) {
    auto &&[subA, idxA] = filterAndGetIdx(orderA, filterA);
    auto &&[subB, idxB] = filterAndGetIdx(orderB, filterB);
    if (subA == subB) {
        return; // OK
    }
    if (auto &&subPermu = permutationB2A(subB, subA); subPermu.has_value()) {
        auto fullPermu =
            ranges::to<std::vector>(views::ints(0, (int)orderB.size()));
        for (auto &&[i, p] : views::enumerate(*subPermu)) {
            fullPermu[idxB[i]] = idxB[p];
        }
        throw NeedVarReorder(idB, fullPermu, message); // Retry
    } else {
        throw InvalidSchedule(message); // Impossible
    }
}

void AsMatMul::checkSameOrderNoRetry(const ID &idA,
                                     const std::vector<int> &orderA,
                                     const std::vector<bool> &filterA,
                                     const ID &idB,
                                     const std::vector<int> &orderB,
                                     const std::vector<bool> &filterB,
                                     const std::string &message) {
    if (filter(orderA, filterA) != filter(orderB, filterB)) {
        throw InvalidSchedule(message);
    }
}

void AsMatMul::retryReorderingBack(const ID &id,
                                   const std::vector<bool> &filter,
                                   const std::string &message) {
    std::vector<int> permu;
    permu.reserve(filter.size());
    for (int i = 0, n = filter.size(); i < n; i++) {
        if (!filter[i]) {
            permu.emplace_back(i);
        }
    }
    for (int i = 0, n = filter.size(); i < n; i++) {
        if (filter[i]) {
            permu.emplace_back(i);
        }
    }
    throw NeedVarReorder{id, permu, message};
}

void AsMatMul::retryReorderingFront(const ID &id,
                                    const std::vector<bool> &filter,
                                    const std::string &message) {
    std::vector<int> permu;
    permu.reserve(filter.size());
    for (int i = 0, n = filter.size(); i < n; i++) {
        if (filter[i]) {
            permu.emplace_back(i);
        }
    }
    for (int i = 0, n = filter.size(); i < n; i++) {
        if (!filter[i]) {
            permu.emplace_back(i);
        }
    }
    throw NeedVarReorder{id, permu, message};
}

const LinearExpr<int64_t> &AsMatMul::analyzeLinear(const Expr &expr) {
    analyzeLinear_(expr);
    return analyzeLinear_.result().at(expr);
}

Stmt AsMatMul::visitStmt(const Stmt &op) {
    if (inside_ && op->nodeType() != ASTNodeType::ReduceTo &&
        op->nodeType() != ASTNodeType::Store &&
        op->nodeType() != ASTNodeType::StmtSeq &&
        op->nodeType() != ASTNodeType::For &&
        op->nodeType() != ASTNodeType::VarDef) {
        throw InvalidSchedule(FT_MSG << "Unexpected " << op->nodeType()
                                     << " node");
    }
    return BaseClass::visitStmt(op);
}

Stmt AsMatMul::visit(const For &op) {
    if (inside_) {
        iterMap_[op->iter_] = nestCnt_++;
        auto ret = BaseClass::visit(op);
        iterMap_.erase(op->iter_), nestCnt_--;
        return ret;
    } else if (op->id() == loop_) {
        inside_ = true;
        iterMap_[op->iter_] = nestCnt_++;
        auto ret = BaseClass::visit(op);
        iterMap_.erase(op->iter_), nestCnt_--;
        inside_ = false;

        Expr alpha, beta;
        if (!foundLeaf_) {
            throw InvalidSchedule("`c += a * b` statement not found");
        }
        alpha = makeIntConst(1);
        if (foundInit_) {
            if (!HashComparator()(c_, initC_)) {
                throw InvalidSchedule(
                    "The initialized matrix " + initC_.as<LoadNode>()->var_ +
                    " does not match " + c_.as<LoadNode>()->var_ +
                    ", the matrix being reduced to");
            }
            beta = makeIntConst(0);
        } else {
            beta = makeIntConst(1);
        }
        ret = makeMatMul(backend_, a_, b_, c_, alpha, beta, m_, k_, n_, lda_,
                         ldb_, ldc_, stridea_, strideb_, stridec_, batchSize_,
                         aIsRowMajor_, bIsRowMajor_, cIsRowMajor_, ret);
        for (auto &&def : innerDefs_) {
            ret = makeVarDef(def->name_, def->buffer_, def->viewOf_, ret,
                             def->pinned_, def->metadata(), def->id());
        }
        done_ = true;
        return ret;
    } else {
        ASSERT(!outerDefs_.count(op->iter_));
        outerDefs_.insert(op->iter_);
        auto ret = BaseClass::visit(op);
        outerDefs_.erase(op->iter_);
        return ret;
    }
}

Stmt AsMatMul::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();

    if (inside_) {
        if (foundLeaf_) {
            throw InvalidSchedule(
                "a Store node should not be after a ReduceTo node");
        }
        foundInit_ = true;

        if (!isConst0(op->expr_)) {
            throw InvalidSchedule("Matrix c can either be not initialized or "
                                  "initialized to zeros");
        }

        std::vector<bool> used;
        std::tie(used, orderInit_, initC_) = findIterUsedAndBaseAddr(op);
    }

    return op;
}

Stmt AsMatMul::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    if (inside_) {
        if (foundLeaf_) {
            throw InvalidSchedule("Unexpected multiple ReduceTo node");
        }
        foundLeaf_ = true;

        if (op->op_ != ReduceOp::Add) {
            throw InvalidSchedule("`+=` not found");
        }

        if (op->expr_->nodeType() != ASTNodeType::Mul) {
            throw InvalidSchedule("Multiplication not found");
        }
        Mul mul = op->expr_.as<MulNode>();

        if (mul->lhs_->nodeType() != ASTNodeType::Load) {
            throw InvalidSchedule("Matrix a not found");
        }
        Load loadA = mul->lhs_.as<LoadNode>();

        if (mul->rhs_->nodeType() != ASTNodeType::Load) {
            throw InvalidSchedule("Matrix b not found");
        }
        Load loadB = mul->rhs_.as<LoadNode>();

        // Find out which LOOPS are used
        std::vector<bool> usedByA, usedByB, usedByC;
        std::vector<int> orderA, orderB, orderC;
        std::tie(usedByA, orderA, a_) = findIterUsedAndBaseAddr(loadA);
        std::tie(usedByB, orderB, b_) = findIterUsedAndBaseAddr(loadB);
        std::tie(usedByC, orderC, c_) = findIterUsedAndBaseAddr(op);
        std::vector<bool> mAxes(nestCnt_, false);
        std::vector<bool> kAxes(nestCnt_, false);
        std::vector<bool> nAxes(nestCnt_, false);
        std::vector<bool> batchAxes(nestCnt_, false);
        for (int i = 0; i < nestCnt_; i++) {
            batchAxes[i] = usedByA[i] && usedByB[i] && usedByC[i];
            mAxes[i] = usedByA[i] && !usedByB[i] && usedByC[i];
            kAxes[i] = usedByA[i] && usedByB[i] && !usedByC[i];
            nAxes[i] = !usedByA[i] && usedByB[i] && usedByC[i];
        }

        ID idA = def(loadA->var_)->id();
        ID idB = def(loadB->var_)->id();
        ID idC = def(op->var_)->id();

        checkSameOrderOrRetry(idA, orderA, batchAxes, idB, orderB, batchAxes,
                              "Order of each indices in the batch axis should "
                              "be the same in each matrices");
        checkSameOrderOrRetry(idA, orderA, batchAxes, idC, orderC, batchAxes,
                              "Order of each indices in the batch axis should "
                              "be the same in each matrices");
        checkSameOrderOrRetry(idA, orderA, mAxes, idC, orderC, mAxes,
                              "Order of each indices in the m axis should be "
                              "the same in each matrices");
        checkSameOrderOrRetry(idA, orderA, kAxes, idB, orderB, kAxes,
                              "Order of each indices in the k axis should be "
                              "the same in each matrices");
        checkSameOrderOrRetry(idB, orderB, nAxes, idC, orderC, nAxes,
                              "Order of each indices in the n axis should be "
                              "the same in each matrices");
        if (foundInit_) {
            checkSameOrderNoRetry(
                idC, orderInit_, batchAxes, idC, orderC, batchAxes,
                "Order of each indices in the batch axis should be the same in "
                "initialization and reduction");
            checkSameOrderNoRetry(
                idC, orderInit_, mAxes, idC, orderC, mAxes,
                "Order of each indices in the m axis should be the same in "
                "initialization and reduction");
            checkSameOrderNoRetry(
                idC, orderInit_, nAxes, idC, orderC, nAxes,
                "Order of each indices in the n axis should be the same in "
                "initialization and reduction");
        }

        // Find out which TENSOR DIMENSIONS are used
        std::vector<bool> dimsABatch = findDimsUsed(loadA, batchAxes);
        std::vector<bool> dimsBBatch = findDimsUsed(loadB, batchAxes);
        std::vector<bool> dimsCBatch = findDimsUsed(op, batchAxes);
        std::vector<bool> dimsAM = findDimsUsed(loadA, mAxes);
        std::vector<bool> dimsAK = findDimsUsed(loadA, kAxes);
        std::vector<bool> dimsBK = findDimsUsed(loadB, kAxes);
        std::vector<bool> dimsBN = findDimsUsed(loadB, nAxes);
        std::vector<bool> dimsCM = findDimsUsed(op, mAxes);
        std::vector<bool> dimsCN = findDimsUsed(op, nAxes);

        Expr strideAM, strideAK, strideBK, strideBN, strideCM, strideCN;
        std::tie(batchSize_, stridea_) = findLenAndStride(loadA, dimsABatch);
        std::tie(batchSize_, strideb_) = findLenAndStride(loadB, dimsBBatch);
        std::tie(batchSize_, stridec_) = findLenAndStride(op, dimsCBatch);
        std::tie(m_, strideAM) = findLenAndStride(loadA, dimsAM);
        std::tie(k_, strideAK) = findLenAndStride(loadA, dimsAK);
        std::tie(k_, strideBK) = findLenAndStride(loadB, dimsBK);
        std::tie(n_, strideBN) = findLenAndStride(loadB, dimsBN);
        std::tie(m_, strideCM) = findLenAndStride(op, dimsCM);
        std::tie(n_, strideCN) = findLenAndStride(op, dimsCN);
        if (isIntConst1(strideAK)) {
            aIsRowMajor_ = true;
            lda_ = strideAM;
        } else if (isIntConst1(strideAM)) {
            aIsRowMajor_ = false;
            lda_ = strideAK;
        } else {
            retryReorderingBack(
                idA, dimsAK,
                "Either m or k dimension of a should be 1-strided");
        }
        if (isIntConst1(strideBN)) {
            bIsRowMajor_ = true;
            ldb_ = strideBK;
        } else if (isIntConst1(strideBK)) {
            bIsRowMajor_ = false;
            ldb_ = strideBN;
        } else {
            retryReorderingBack(
                idB, dimsBN,
                "Either k or n dimension of b should be 1-strided");
        }
        if (isIntConst1(strideCN)) {
            cIsRowMajor_ = true;
            ldc_ = strideCM;
        } else if (isIntConst1(strideCM)) {
            cIsRowMajor_ = false;
            ldc_ = strideCN;
        } else {
            retryReorderingBack(
                idC, dimsCN,
                "Either m or n dimension of c should be 1-strided");
        }

        // Fix the strides of singleton dimensions to satisfy the API
        // requirement
        if (std::find(mAxes.begin(), mAxes.end(), true) == mAxes.end()) {
            lda_ = aIsRowMajor_ ? k_ : m_;
            ldc_ = cIsRowMajor_ ? n_ : m_;
        }
        if (std::find(kAxes.begin(), kAxes.end(), true) == kAxes.end()) {
            lda_ = aIsRowMajor_ ? k_ : m_;
            ldb_ = bIsRowMajor_ ? n_ : k_;
        }
        if (std::find(nAxes.begin(), nAxes.end(), true) == nAxes.end()) {
            ldb_ = bIsRowMajor_ ? n_ : k_;
            ldc_ = cIsRowMajor_ ? n_ : m_;
        }
        if (std::find(batchAxes.begin(), batchAxes.end(), true) ==
            batchAxes.end()) { // There is no batch axes
            stridea_ = makeMul(lda_, aIsRowMajor_ ? m_ : k_);
            strideb_ = makeMul(ldb_, bIsRowMajor_ ? k_ : n_);
            stridec_ = makeMul(ldc_, cIsRowMajor_ ? m_ : n_);
        } else {
            if (!inOrder(dimsABatch, dimsAM) || !inOrder(dimsABatch, dimsAK)) {
                retryReorderingFront(idA, dimsABatch,
                                     "BLAS requires batch dimensions to be out "
                                     "of matrix dimensions in A");
            }
            if (!inOrder(dimsBBatch, dimsBK) || !inOrder(dimsABatch, dimsBN)) {
                retryReorderingFront(idB, dimsBBatch,
                                     "BLAS requires batch dimensions to be out "
                                     "of matrix dimensions in B");
            }
            if (!inOrder(dimsCBatch, dimsCM) || !inOrder(dimsABatch, dimsCN)) {
                retryReorderingFront(idC, dimsCBatch,
                                     "BLAS requires batch dimensions to be out "
                                     "of matrix dimensions in C");
            }
        }
    }

    return op;
}

Stmt AsMatMul::visit(const VarDef &op) {
    ASSERT(!outerDefs_.count(op->name_));
    outerDefs_.insert(op->name_);
    auto ret = BaseClass::visit(op);
    outerDefs_.erase(op->name_);
    if (inside_) {
        innerDefs_.emplace_back(op);
        return op->body_;
    } else {
        return ret;
    }
}

Stmt asMatMul(const Stmt &_ast, const ID &loop, MatMulBackend backend) {
    AsMatMul mutator(loop, backend);
    auto ast = simplify(_ast); // Simplify confusing loop range and indexing
                               // from libop. TODO: simplify only needed region
    ast = mutator(ast);
    if (!mutator.done()) {
        throw InvalidSchedule(FT_MSG << loop << " not found");
    }
    return ast;
}

void Schedule::asMatMul(const ID &loop, AsMatMulMode mode,
                        const Ref<Target> &target, MatMulBackend backend) {
    beginTransaction();
    while (true) {
        auto log = appendLog(
            MAKE_SCHEDULE_LOG(AsMatMul, freetensor::asMatMul, loop, backend));
        try {
            applyLog(log);
            break;
        } catch (const NeedVarReorder &e) {
            if (mode != AsMatMulMode::KeepMemLayout) {
                try {
                    ID defId = e.vardef_;
                    if (mode == AsMatMulMode::TryTranspose) {
                        auto def = find(defId).as<VarDefNode>();
                        defId = std::get<3>(
                            cache(loop, def->name_, def->buffer_->mtype()));
                    }
                    varReorder(defId, e.order_);
                } catch (const InvalidSchedule &e2) {
                    abortTransaction();
                    throw InvalidSchedule(
                        log, ast(),
                        FT_MSG << e.what()
                               << ". Tried var_reorder, but resulting in "
                                  "another exception: "
                               << e2.what());
                }
            } else {
                abortTransaction();
                throw InvalidSchedule(log, ast(), e.what());
            }
        } catch (const InvalidSchedule &e) {
            abortTransaction();
            throw InvalidSchedule(log, ast(), e.what());
        }
    }
    commitTransaction();
}

void Schedule::asMatMul(const ID &loop, AsMatMulMode mode,
                        const Ref<Target> &target) {
    switch (target->type()) {
    case TargetType::CPU:
        asMatMul(loop, mode, target, MatMulBackend::Mkl);
        break;
    case TargetType::GPU:
        asMatMul(loop, mode, target, MatMulBackend::Cutlass);
        break;
    default:
        throw InvalidSchedule(
            ast(), FT_MSG << "No default MatMul backend for target " << target);
    }
}

void Schedule::asMatMul(const ID &loop, AsMatMulMode mode) {
    asMatMul(loop, mode, Config::defaultTarget());
}

} // namespace freetensor
