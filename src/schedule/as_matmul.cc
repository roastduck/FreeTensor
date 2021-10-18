#include <pass/make_reduction.h>
#include <pass/simplify.h>
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
            if (getHash(c_) != getHash(initC_)) {
                throw InvalidSchedule(
                    "The initialized matrix " + initC_.as<LoadNode>()->var_ +
                    " does not match " + c_.as<LoadNode>()->var_ +
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
        ASSERT(!outerDefs_.count(op->iter_));
        outerDefs_.insert(op->iter_);
        auto ret = Mutator::visit(op);
        outerDefs_.erase(op->iter_);
        return ret;
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

        std::vector<bool> used;
        std::tie(used, orderInit_, initC_) = findIterUsedAndBaseAddr(op);
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

        if (mul->rhs_->nodeType() != ASTNodeType::Load) {
            throw InvalidSchedule("Matrix b not found");
        }
        Load loadB = mul->rhs_.as<LoadNode>();

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

        if (filter(orderA, batchAxes) != filter(orderB, batchAxes) ||
            filter(orderA, batchAxes) != filter(orderC, batchAxes)) {
            throw InvalidSchedule("Order of each indices in the batch axis "
                                  "should be the same in each matrices");
        }
        if (filter(orderA, mAxes) != filter(orderC, mAxes)) {
            throw InvalidSchedule("Order of each indices in the m axis "
                                  "should be the same in each matrices");
        }
        if (filter(orderA, kAxes) != filter(orderB, kAxes)) {
            throw InvalidSchedule("Order of each indices in the k axis "
                                  "should be the same in each matrices");
        }
        if (filter(orderB, nAxes) != filter(orderC, nAxes)) {
            throw InvalidSchedule("Order of each indices in the n axis "
                                  "should be the same in each matrices");
        }
        if (foundInit_) {
            if (filter(orderInit_, batchAxes) != filter(orderC, batchAxes)) {
                throw InvalidSchedule(
                    "Order of each indices in the batch axis "
                    "should be the same in initialization and reduction");
            }
            if (filter(orderInit_, mAxes) != filter(orderC, mAxes)) {
                throw InvalidSchedule(
                    "Order of each indices in the m axis "
                    "should be the same in initialization and reduction");
            }
            if (filter(orderInit_, nAxes) != filter(orderC, nAxes)) {
                throw InvalidSchedule(
                    "Order of each indices in the n axis "
                    "should be the same in initialization and reduction");
            }
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
            batchAxes.end()) {
            stridea_ = makeMul(lda_, aIsRowMajor_ ? m_ : k_);
            strideb_ = makeMul(ldb_, bIsRowMajor_ ? k_ : n_);
            stridec_ = makeMul(ldc_, cIsRowMajor_ ? m_ : n_);
        }
    }

    return op;
}

Stmt AsMatMul::visit(const VarDef &op) {
    ASSERT(!buffers_.count(op->name_));
    ASSERT(!outerDefs_.count(op->name_));
    buffers_[op->name_] = op->buffer_;
    outerDefs_.insert(op->name_);
    auto ret = Mutator::visit(op);
    outerDefs_.erase(op->name_);
    buffers_.erase(op->name_);
    return ret;
}

Stmt asMatMul(const Stmt &_ast, const std::string &loop) {
    auto ast = simplifyPass(_ast); // const prop
    ast = makeReduction(ast);
    ast = AsMatMul(loop)(ast);
    return ast;
}

} // namespace ir
