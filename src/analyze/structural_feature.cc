#include <climits>

#include <analyze/check_all_defined.h>
#include <analyze/structural_feature.h>

namespace ir {

void StructuralFeature::updCompInfo(const AST &parent, const AST &child,
                                    int repeat) {
    for (auto &&item : info_[child].opCnt_) {
        if (info_[parent].opCnt_[item.first] == -1 || item.second == -1 ||
            repeat == -1) {
            info_[parent].opCnt_[item.first] = -1;
        } else {
            info_[parent].opCnt_[item.first] += item.second * repeat;
        }
    }
}

void StructuralFeature::updAccCntInfo(const AST &parent, const AST &child,
                                      int repeat) {
    for (auto &&item : info_[child].loadCnt_) {
        if (info_[parent].loadCnt_[item.first] == -1 || item.second == -1 ||
            repeat == -1) {
            info_[parent].loadCnt_[item.first] = -1;
        } else {
            info_[parent].loadCnt_[item.first] += item.second * repeat;
        }
    }
    for (auto &&item : info_[child].storeCnt_) {
        if (info_[parent].storeCnt_[item.first] == -1 || item.second == -1 ||
            repeat == -1) {
            info_[parent].storeCnt_[item.first] = -1;
        } else {
            info_[parent].storeCnt_[item.first] += item.second * repeat;
        }
    }
    for (auto &&item : info_[child].accessCnt_) {
        if (info_[parent].accessCnt_[item.first] == -1 || item.second == -1 ||
            repeat == -1) {
            info_[parent].accessCnt_[item.first] = -1;
        } else {
            info_[parent].accessCnt_[item.first] += item.second * repeat;
        }
    }
}

void StructuralFeature::updAreaInfo(const AST &parent, const AST &child) {
    auto filter = [this](NodeBufferInfo &child) -> NodeBufferInfo {
        size_t n = child.lo_.size();
        ASSERT(child.hi_.size() == n);

        NodeBufferInfo ret;
        ret.lo_ = std::vector<LowerBoundsList>(n);
        ret.hi_ = std::vector<UpperBoundsList>(n);
        for (size_t i = 0; i < n; i++) {
            for (auto &&b : child.lo_[i]) {
                if (checkAllDefined(names(), b.expr())) {
                    ret.lo_[i].emplace_back(b);
                }
            }
            for (auto &&b : child.hi_[i]) {
                if (checkAllDefined(names(), b.expr())) {
                    ret.hi_[i].emplace_back(b);
                }
            }
        }
        return ret;
    };

    auto merge = [](const NodeBufferInfo &parent,
                    const NodeBufferInfo &child) -> NodeBufferInfo {
        size_t n = parent.lo_.size();
        ASSERT(parent.hi_.size() == n);
        ASSERT(child.lo_.size() == n);
        ASSERT(child.hi_.size() == n);

        NodeBufferInfo ret;
        ret.lo_ = std::vector<LowerBoundsList>(n);
        ret.hi_ = std::vector<UpperBoundsList>(n);
        for (size_t i = 0; i < n; i++) {
            for (auto &&b1 : parent.lo_[i]) {
                for (auto &&b2 : child.lo_[i]) {
                    if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                        ret.lo_[i].emplace_back(LinearExpr<Rational<int64_t>>{
                            {}, std::min(b1.lin().bias_, b2.lin().bias_)});
                    }
                }
            }
            for (auto &&b1 : parent.hi_[i]) {
                for (auto &&b2 : child.hi_[i]) {
                    if (b1.lin().coeff_.empty() && b2.lin().coeff_.empty()) {
                        ret.hi_[i].emplace_back(LinearExpr<Rational<int64_t>>{
                            {}, std::max(b1.lin().bias_, b2.lin().bias_)});
                    }
                }
            }
        }
        return ret;
    };

    for (auto &&buf : info_[child].loads_) {
        if (!info_[parent].loads_.count(buf.first)) {
            info_[parent].loads_[buf.first] = filter(buf.second);
        } else {
            info_[parent].loads_[buf.first] =
                merge(info_[parent].loads_[buf.first], filter(buf.second));
        }
    }
    for (auto &&buf : info_[child].stores_) {
        if (!info_[parent].stores_.count(buf.first)) {
            info_[parent].stores_[buf.first] = filter(buf.second);
        } else {
            info_[parent].stores_[buf.first] =
                merge(info_[parent].stores_[buf.first], filter(buf.second));
        }
    }
    for (auto &&buf : info_[child].accesses_) {
        if (!info_[parent].accesses_.count(buf.first)) {
            info_[parent].accesses_[buf.first] = filter(buf.second);
        } else {
            info_[parent].accesses_[buf.first] =
                merge(info_[parent].accesses_[buf.first], filter(buf.second));
        }
    }

    for (auto &&item : info_[child].innerLoadArea_) {
        info_[parent].innerLoadArea_[item.first] += item.second;
    }
    for (auto &&item : info_[child].innerStoreArea_) {
        info_[parent].innerStoreArea_[item.first] += item.second;
    }
    for (auto &&item : info_[child].innerAccessArea_) {
        info_[parent].innerAccessArea_[item.first] += item.second;
    }
}

void StructuralFeature::updInfo(const AST &parent, const AST &child,
                                int repeat) {
    updCompInfo(parent, child, repeat);
    updAccCntInfo(parent, child, repeat);
    updAreaInfo(parent, child);
}

void StructuralFeature::calcCompFeatures(const Stmt &node) {
    features_[node->id()].opCnt_ = info_[node].opCnt_;
}

void StructuralFeature::calcAccCntFeatures(const Stmt &node) {
    features_[node->id()].loadCnt_ = info_[node].loadCnt_;
    features_[node->id()].storeCnt_ = info_[node].storeCnt_;
    features_[node->id()].accessCnt_ = info_[node].accessCnt_;
}

int64_t StructuralFeature::calcArea(const NodeBufferInfo &bufInfo) {
    size_t n = bufInfo.lo_.size();
    ASSERT(bufInfo.hi_.size() == n);
    int64_t area = 1;
    for (size_t i = 0; i < n; i++) {
        int64_t tightest = LLONG_MAX;
        for (auto &&lo : bufInfo.lo_[i]) {
            for (auto &&hi : bufInfo.hi_[i]) {
                auto diff = sub(hi, lo);
                if (diff.lin().coeff_.empty()) {
                    tightest =
                        std::min(tightest,
                                 diff.lin().bias_.p_ / diff.lin().bias_.q_ + 1);
                }
            }
        }
        if (tightest < LLONG_MAX) {
            area *= tightest;
        }
    }
    return area;
}

void StructuralFeature::calcAreaFeatures(const Stmt &node) {
    for (auto &&item : info_[node].innerLoadArea_) {
        features_[node->id()].loadArea_[item.first] = item.second;
    }
    for (auto &&item : info_[node].innerStoreArea_) {
        features_[node->id()].storeArea_[item.first] = item.second;
    }
    for (auto &&item : info_[node].innerAccessArea_) {
        features_[node->id()].accessArea_[item.first] = item.second;
    }

    for (auto &&item : info_[node].loads_) {
        features_[node->id()].loadArea_[buffer(item.first)->mtype()] +=
            calcArea(item.second);
    }
    for (auto &&item : info_[node].stores_) {
        features_[node->id()].storeArea_[buffer(item.first)->mtype()] +=
            calcArea(item.second);
    }
    for (auto &&item : info_[node].accesses_) {
        features_[node->id()].accessArea_[buffer(item.first)->mtype()] +=
            calcArea(item.second);
    }
}

void StructuralFeature::calcFeatures(const Stmt &node) {
    calcCompFeatures(node);
    calcAccCntFeatures(node);
    calcAreaFeatures(node);
}

Expr StructuralFeature::visitBinOp(const BinaryExpr &_op) {
    auto __op = BaseClass::visitExpr(_op);
    ASSERT(__op->nodeType() == _op->nodeType());
    auto op = __op.as<BinaryExprNode>();
    updInfo(op, op->lhs_);
    updInfo(op, op->rhs_);
    info_[op].opCnt_[upCast(dtype(op->lhs_), dtype(op->rhs_))]++;
    return op;
}

Expr StructuralFeature::visitUnaryOp(const UnaryExpr &_op) {
    auto __op = BaseClass::visitExpr(_op);
    ASSERT(__op->nodeType() == _op->nodeType());
    auto op = __op.as<UnaryExprNode>();
    updInfo(op, op->expr_);
    info_[op].opCnt_[dtype(op->expr_)]++;
    return op;
}

Stmt StructuralFeature::visitStmt(const Stmt &_op) {
    auto op = BaseClass::visitStmt(_op);
    calcFeatures(op);
    return op;
}

Expr StructuralFeature::visitExpr(const Expr &op) {
    if (op->isBinary()) {
        return visitBinOp(op.as<BinaryExprNode>());
    } else if (op->isUnary()) {
        return visitUnaryOp(op.as<UnaryExprNode>());
    } else {
        return BaseClass::visitExpr(op);
    }
}

Expr StructuralFeature::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();

    NodeBufferInfo &loads = info_[op].loads_[op->var_];
    NodeBufferInfo &accesses = info_[op].accesses_[op->var_];
    loads.lo_.reserve(op->indices_.size());
    loads.hi_.reserve(op->indices_.size());
    accesses.lo_.reserve(op->indices_.size());
    accesses.hi_.reserve(op->indices_.size());
    for (auto &&idx : op->indices_) {
        loads.lo_.emplace_back(getLower(idx));
        loads.hi_.emplace_back(getUpper(idx));
        accesses.lo_.emplace_back(getLower(idx));
        accesses.hi_.emplace_back(getUpper(idx));
    }

    info_[op].loadCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }

    return op;
}

Stmt StructuralFeature::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    auto op = __op.as<StoreNode>();

    NodeBufferInfo &stores = info_[op].stores_[op->var_];
    NodeBufferInfo &accesses = info_[op].accesses_[op->var_];
    stores.lo_.reserve(op->indices_.size());
    stores.hi_.reserve(op->indices_.size());
    accesses.lo_.reserve(op->indices_.size());
    accesses.hi_.reserve(op->indices_.size());
    for (auto &&idx : op->indices_) {
        stores.lo_.emplace_back(getLower(idx));
        stores.hi_.emplace_back(getUpper(idx));
        accesses.lo_.emplace_back(getLower(idx));
        accesses.hi_.emplace_back(getUpper(idx));
    }

    info_[op].storeCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }
    updInfo(op, op->expr_);

    return op;
}

Stmt StructuralFeature::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    auto op = __op.as<ReduceToNode>();

    NodeBufferInfo &loads = info_[op].loads_[op->var_];
    NodeBufferInfo &stores = info_[op].stores_[op->var_];
    NodeBufferInfo &accesses = info_[op].accesses_[op->var_];
    loads.lo_.reserve(op->indices_.size());
    loads.hi_.reserve(op->indices_.size());
    stores.lo_.reserve(op->indices_.size());
    stores.hi_.reserve(op->indices_.size());
    accesses.lo_.reserve(op->indices_.size());
    accesses.hi_.reserve(op->indices_.size());
    for (auto &&idx : op->indices_) {
        loads.lo_.emplace_back(getLower(idx));
        loads.hi_.emplace_back(getUpper(idx));
        stores.lo_.emplace_back(getLower(idx));
        stores.hi_.emplace_back(getUpper(idx));
        accesses.lo_.emplace_back(getLower(idx));
        accesses.hi_.emplace_back(getUpper(idx));
    }

    info_[op]
        .opCnt_[upCast(buffer(op->var_)->tensor().dtype(), dtype(op->expr_))]++;
    info_[op].loadCnt_[buffer(op->var_)->mtype()]++;
    info_[op].storeCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }
    updInfo(op, op->expr_);

    return op;
}

Expr StructuralFeature::visit(const IfExpr &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::IfExpr);
    auto op = __op.as<IfExprNode>();
    updInfo(op, op->cond_);
    updInfo(op, op->thenCase_);
    updInfo(op, op->elseCase_);
    info_[op].opCnt_[dtype(op->cond_)]++;
    return op;
}

Expr StructuralFeature::visit(const Cast &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Cast);
    auto op = __op.as<CastNode>();
    updInfo(op, op->expr_);
    info_[op].opCnt_[dtype(op->expr_)]++;
    return op;
}

Stmt StructuralFeature::visit(const StmtSeq &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::StmtSeq);
    auto op = __op.as<StmtSeqNode>();

    for (auto &&stmt : op->stmts_) {
        updInfo(op, stmt);
    }

    return op;
}

Stmt StructuralFeature::visit(const If &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::If);
    auto op = __op.as<IfNode>();

    updInfo(op, op->cond_);
    updInfo(op, op->thenCase_);
    if (op->elseCase_.isValid()) {
        updInfo(op, op->elseCase_);
    }

    return op;
}

Stmt StructuralFeature::visit(const Assert &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Assert);
    auto op = __op.as<AssertNode>();

    updInfo(op, op->cond_);
    updInfo(op, op->body_);

    return op;
}

Stmt StructuralFeature::visit(const For &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();

    updInfo(op, op->begin_);
    updInfo(op, op->end_);
    if (auto intLen = getInt(op->len_); intLen.isValid()) {
        updInfo(op, op->body_, *intLen);
    } else {
        updInfo(op, op->body_, -1);
    }

    return op;
}

Stmt StructuralFeature::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();

    for (auto &&item : op->buffer_->tensor().shape()) {
        updInfo(op, item);
    }
    updInfo(op, op->body_);

    if (info_[op].loads_.count(op->name_)) {
        info_[op].innerLoadArea_[op->buffer_->mtype()] +=
            calcArea(info_[op].loads_[op->name_]);
        info_[op].loads_.erase(op->name_);
    }
    if (info_[op].stores_.count(op->name_)) {
        info_[op].innerStoreArea_[op->buffer_->mtype()] +=
            calcArea(info_[op].stores_[op->name_]);
        info_[op].stores_.erase(op->name_);
    }
    if (info_[op].accesses_.count(op->name_)) {
        info_[op].innerAccessArea_[op->buffer_->mtype()] +=
            calcArea(info_[op].accesses_[op->name_]);
        info_[op].accesses_.erase(op->name_);
    }

    return op;
}

} // namespace ir
