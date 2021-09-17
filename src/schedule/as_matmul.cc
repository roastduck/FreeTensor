#include <schedule/as_matmul.h>

namespace ir {

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

uint64_t AsMatMul::getHash(const Expr &op) {
    getHash_(op);
    return getHash_.hash().at(op);
}

Stmt AsMatMul::visitStmt(const Stmt &op,
                         const std::function<Stmt(const Stmt &)> &visitNode) {
    if (inside_ && op->nodeType() != ASTNodeType::ReduceTo &&
        op->nodeType() != ASTNodeType::Store &&
        op->nodeType() != ASTNodeType::StmtSeq &&
        op->nodeType() != ASTNodeType::For) {
        throw InvalidSchedule("Unexpected " + toString(op->nodeType()) +
                              " node");
    }
    return Mutator::visitStmt(op, visitNode);
}

Stmt AsMatMul::visit(const For &op) {
    if (inside_) {
        iterMap_[op->iter_] = nestCnt_++;
        nests_.emplace_back(op);
        auto ret = Mutator::visit(op);
        nests_.pop_back();
        iterMap_.erase(op->iter_), nestCnt_--;
        return ret;
    } else if (op->id() == loop_) {
        inside_ = true;
        iterMap_[op->iter_] = nestCnt_++;
        nests_.emplace_back(op);
        auto ret = Mutator::visit(op);
        nests_.pop_back();
        iterMap_.erase(op->iter_), nestCnt_--;
        inside_ = false;

        Expr alpha, beta;
        if (!foundLeaf_) {
            throw InvalidSchedule("`c += a * b` statement not found");
        }
        alpha = makeIntConst(1);
        if (foundInit_) {
            if (c_ != initC_) {
                throw InvalidSchedule("The initialized matrix " + initC_ +
                                      " does not match " + c_ +
                                      ", the matrix being reduced to");
            }
            beta = makeIntConst(0);
        } else {
            beta = makeIntConst(1);
        }
        return makeMatMul("", a_, b_, c_, alpha, beta, m_, k_, n_, lda_, ldb_,
                          ldc_, stridea_, strideb_, stridec_, batchSize_,
                          aIsRowMajor_, bIsRowMajor_, cIsRowMajor_, ret);
    } else {
        return Mutator::visit(op);
    }
}

Stmt AsMatMul::visit(const Store &_op) {
    auto __op = Mutator::visit(_op);
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

        std::vector<bool> used = countIterUsed(op);
        if (std::find(used.begin(), used.end(), false) != used.end()) {
            throw InvalidSchedule("At least one dimension is not initialized");
        }

        initC_ = op->var_;
    }

    return op;
}

Stmt AsMatMul::visit(const ReduceTo &_op) {
    auto __op = Mutator::visit(_op);
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
        a_ = loadA->var_;

        if (mul->rhs_->nodeType() != ASTNodeType::Load) {
            throw InvalidSchedule("Matrix b not found");
        }
        Load loadB = mul->rhs_.as<LoadNode>();
        b_ = loadB->var_;

        c_ = op->var_;

        std::vector<bool> usedByA = countIterUsed(loadA);
        std::vector<bool> usedByB = countIterUsed(loadB);
        std::vector<bool> usedByC = countIterUsed(op);
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

        Expr strideAM, strideAK, strideBK, strideBN, strideCM, strideCN;
        std::tie(batchSize_, stridea_) = findLenAndStride(loadA, batchAxes);
        std::tie(batchSize_, strideb_) = findLenAndStride(loadB, batchAxes);
        std::tie(batchSize_, stridec_) = findLenAndStride(op, batchAxes);
        std::tie(m_, strideAM) = findLenAndStride(loadA, mAxes);
        std::tie(k_, strideAK) = findLenAndStride(loadA, kAxes);
        std::tie(k_, strideBK) = findLenAndStride(loadB, kAxes);
        std::tie(n_, strideBN) = findLenAndStride(loadB, nAxes);
        std::tie(m_, strideCM) = findLenAndStride(op, mAxes);
        std::tie(n_, strideCN) = findLenAndStride(op, nAxes);
        if (isIntConst1(strideAK)) {
            aIsRowMajor_ = true;
            lda_ = strideAM;
        } else if (isIntConst1(strideAM)) {
            aIsRowMajor_ = false;
            lda_ = strideAK;
        } else {
            throw InvalidSchedule(
                "Eiter m or k dimension of a should be 1-strided");
        }
        if (isIntConst1(strideBN)) {
            bIsRowMajor_ = true;
            ldb_ = strideBK;
        } else if (isIntConst1(strideBK)) {
            bIsRowMajor_ = false;
            ldb_ = strideBN;
        } else {
            throw InvalidSchedule(
                "Eiter k or n dimension of b should be 1-strided");
        }
        if (isIntConst1(strideCN)) {
            cIsRowMajor_ = true;
            ldc_ = strideCM;
        } else if (isIntConst1(strideCM)) {
            cIsRowMajor_ = false;
            ldc_ = strideCN;
        } else {
            throw InvalidSchedule(
                "Eiter m or n dimension of c should be 1-strided");
        }

        // If not a batched gemm, fix the strides to satisfy the API requirement
        if (std::find(batchAxes.begin(), batchAxes.end(), true) ==
            batchAxes.end()) {
            batchSize_ = makeIntConst(1);
            stridea_ = makeMul(lda_, aIsRowMajor_ ? m_ : k_);
            strideb_ = makeMul(ldb_, bIsRowMajor_ ? k_ : n_);
            stridec_ = makeMul(ldc_, cIsRowMajor_ ? m_ : n_);
        }
    }

    return op;
}

Stmt AsMatMul::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    auto ret = Mutator::visit(op);
    buffers_.erase(op->name_);
    return ret;
}

} // namespace ir
