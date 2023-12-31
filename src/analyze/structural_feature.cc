#include <climits>

#include <analyze/check_all_defined.h>
#include <analyze/comp_unique_bounds_combination.h>
#include <analyze/structural_feature.h>
#include <container_utils.h>
#include <math/min_max.h>

namespace freetensor {

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
    for (auto &&[var, accesses] : info_[child].loads_) {
        if (hasDef(var)) {
            if (!info_[parent].loads_.count(var)) {
                info_[parent].loads_[var] = accesses;
            } else {
                info_[parent].loads_[var] =
                    cat(info_[parent].loads_[var], accesses);
            }
        }
    }
    for (auto &&[var, accesses] : info_[child].stores_) {
        if (hasDef(var)) {
            if (!info_[parent].stores_.count(var)) {
                info_[parent].stores_[var] = accesses;
            } else {
                info_[parent].stores_[var] =
                    cat(info_[parent].stores_[var], accesses);
            }
        }
    }
    for (auto &&[var, accesses] : info_[child].accesses_) {
        if (hasDef(var)) {
            if (!info_[parent].accesses_.count(var)) {
                info_[parent].accesses_[var] = accesses;
            } else {
                info_[parent].accesses_[var] =
                    cat(info_[parent].accesses_[var], accesses);
            }
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

int64_t StructuralFeature::calcArea(
    const std::string &var,
    const std::vector<CompAccessBound::Access> &accesses) {
    if (accesses.empty()) {
        return 0;
    }

    int64_t area = 1;
    size_t n = accesses.front().indices_.size();
    for (size_t i = 0; i < n; i++) {
        // union the bounds of all accesses and get the lower and upper
        // expression
        auto [l, u] = bound_->unionBounds(
            // get bounds of the i-th dimension
            accesses | views::transform([&](auto &&a) {
                return a.bounds_[i]->restrictScope(names());
            }) |
            // ... and pack into vector
            ranges::to<std::vector>());
        l = makeMax(l, makeIntConst(0));
        u = makeMin(
            u, makeSub(buffer(var)->tensor()->shape()[i], makeIntConst(1)));
        auto len = makeAdd(makeSub(u, l), makeIntConst(1));
        if (auto constLen = bound_->getInt(len); constLen.has_value()) {
            area *= *constLen;
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

    for (auto &&[var, accesses] : info_[node].loads_) {
        features_[node->id()].loadArea_[buffer(var)->mtype()] +=
            calcArea(var, accesses);
    }
    for (auto &&[var, accesses] : info_[node].stores_) {
        features_[node->id()].storeArea_[buffer(var)->mtype()] +=
            calcArea(var, accesses);
    }
    for (auto &&[var, accesses] : info_[node].accesses_) {
        features_[node->id()].accessArea_[buffer(var)->mtype()] +=
            calcArea(var, accesses);
    }
}

void StructuralFeature::calcFeatures(const Stmt &node) {
    calcCompFeatures(node);
    calcAccCntFeatures(node);
    calcAreaFeatures(node);
}

void StructuralFeature::visitBinOp(const BinaryExpr &op) {
    BaseClass::visitExpr(op);
    updInfo(op, op->lhs_);
    updInfo(op, op->rhs_);
    info_[op]
        .opCnt_[upCast(op->lhs_->dtype().base(), op->rhs_->dtype().base())]++;
}

void StructuralFeature::visitUnaryOp(const UnaryExpr &op) {
    BaseClass::visitExpr(op);
    updInfo(op, op->expr_);
    info_[op].opCnt_[op->expr_->dtype().base()]++;
}

void StructuralFeature::visitStmt(const Stmt &op) {
    auto boundOfOuterStmt = bound_;
    bound_ = Ref<CompUniqueBoundsCombination>::make(*this);
    BaseClass::visitStmt(op);
    calcFeatures(op);
    bound_ = boundOfOuterStmt;
}

void StructuralFeature::visitExpr(const Expr &op) {
    if (op->isBinary()) {
        visitBinOp(op.as<BinaryExprNode>());
    } else if (op->isUnary()) {
        visitUnaryOp(op.as<UnaryExprNode>());
    } else {
        BaseClass::visitExpr(op);
    }
}

void StructuralFeature::visit(const Load &op) {
    BaseClass::visit(op);

    info_[op].loadCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;
    info_[op].loads_[op->var_] = info_[op].accesses_[op->var_] = {
        CompAccessBound::Access(*bound_, op->indices_, conds())};

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }
}

void StructuralFeature::visit(const Store &op) {
    BaseClass::visit(op);

    info_[op].storeCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;
    info_[op].stores_[op->var_] = info_[op].accesses_[op->var_] = {
        CompAccessBound::Access(*bound_, op->indices_, conds())};

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }
    updInfo(op, op->expr_);
}

void StructuralFeature::visit(const ReduceTo &op) {
    BaseClass::visit(op);

    info_[op].opCnt_[upCast(buffer(op->var_)->tensor()->dtype().base(),
                            op->expr_->dtype().base())]++;
    info_[op].loadCnt_[buffer(op->var_)->mtype()]++;
    info_[op].storeCnt_[buffer(op->var_)->mtype()]++;
    info_[op].accessCnt_[buffer(op->var_)->mtype()]++;
    info_[op].loads_[op->var_] = info_[op].stores_[op->var_] =
        info_[op].accesses_[op->var_] = {
            CompAccessBound::Access(*bound_, op->indices_, conds())};

    for (auto &&idx : op->indices_) {
        updInfo(op, idx);
    }
    updInfo(op, op->expr_);
}

void StructuralFeature::visit(const IfExpr &op) {
    BaseClass::visit(op);
    updInfo(op, op->cond_);
    updInfo(op, op->thenCase_);
    updInfo(op, op->elseCase_);
    info_[op].opCnt_[op->cond_->dtype().base()]++;
}

void StructuralFeature::visit(const Cast &op) {
    BaseClass::visit(op);
    updInfo(op, op->expr_);
    info_[op].opCnt_[op->expr_->dtype().base()]++;
}

void StructuralFeature::visit(const StmtSeq &op) {
    BaseClass::visit(op);
    for (auto &&stmt : op->stmts_) {
        updInfo(op, stmt);
    }
}

void StructuralFeature::visit(const If &op) {
    BaseClass::visit(op);
    updInfo(op, op->cond_);
    updInfo(op, op->thenCase_);
    if (op->elseCase_.isValid()) {
        updInfo(op, op->elseCase_);
    }
}

void StructuralFeature::visit(const Assert &op) {
    BaseClass::visit(op);
    updInfo(op, op->cond_);
    updInfo(op, op->body_);
}

void StructuralFeature::visit(const For &op) {
    BaseClass::visit(op);

    updInfo(op, op->begin_);
    updInfo(op, op->end_);
    if (auto intLen = bound_->getInt(op->len_); intLen.has_value()) {
        updInfo(op, op->body_, *intLen);
    } else {
        updInfo(op, op->body_, -1);
    }
}

void StructuralFeature::visit(const VarDef &op) {
    BaseClass::visit(op);

    for (auto &&item : op->buffer_->tensor()->shape()) {
        updInfo(op, item);
    }
    updInfo(op, op->body_);

    if (info_[op].loads_.count(op->name_)) {
        info_[op].innerLoadArea_[op->buffer_->mtype()] +=
            calcArea(op->name_, info_[op].loads_[op->name_]);
        info_[op].loads_.erase(op->name_);
    }
    if (info_[op].stores_.count(op->name_)) {
        info_[op].innerStoreArea_[op->buffer_->mtype()] +=
            calcArea(op->name_, info_[op].stores_[op->name_]);
        info_[op].stores_.erase(op->name_);
    }
    if (info_[op].accesses_.count(op->name_)) {
        info_[op].innerAccessArea_[op->buffer_->mtype()] +=
            calcArea(op->name_, info_[op].accesses_[op->name_]);
        info_[op].accesses_.erase(op->name_);
    }
}

} // namespace freetensor
