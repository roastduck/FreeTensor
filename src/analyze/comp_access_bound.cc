#include <analyze/check_all_defined.h>
#include <analyze/comp_access_bound.h>

namespace ir {

Expr CompAccessBound::reduceMin(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMin(reduction, item) : item;
}

Expr CompAccessBound::reduceMax(const Expr &reduction, const Expr &item) {
    return reduction.isValid() ? makeMax(reduction, item) : item;
}

Stmt CompAccessBound::visit(const VarDef &_op) {
    access_.erase(_op->name_);

    defs_.insert(_op->name_);
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);
    auto op = __op.as<VarDefNode>();
    defs_.erase(_op->name_);

    if (op->buffer_->atype() != AccessType::Cache) {
        return op;
    }

    if (access_.count(op->name_)) {
        auto &&access = access_.at(op->name_);

        size_t n = op->buffer_->tensor().shape().size();
        op->info_acc_lower_ = Ref<std::vector<Expr>>::make();
        op->info_acc_len_ = Ref<std::vector<Expr>>::make();
        op->info_acc_lower_->reserve(n);
        op->info_acc_len_->reserve(n);

        for (size_t i = 0; i < n; i++) {
            Expr lower, upper;
            for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
                ASSERT(access[j].size() == n);
                auto &&index = access[j][i];
                if (lower_.count(index.get())) {
                    Expr lowerItem;
                    for (auto &&item : lower_.at(index.get())) {
                        if (checkAllDefined(defs_, item)) {
                            lowerItem = reduceMax(lowerItem, item);
                        }
                    }
                    lower = reduceMin(lower, lowerItem);
                } else {
                    lower = makeIntConst(0);
                    break;
                }
            }

            for (size_t j = 0, jEnd = access.size(); j < jEnd; j++) {
                ASSERT(access[j].size() == n);
                auto &&index = access[j][i];
                if (upper_.count(index.get())) {
                    Expr upperItem;
                    for (auto &&item : upper_.at(index.get())) {
                        if (checkAllDefined(defs_, item)) {
                            upperItem = reduceMin(upperItem, item);
                        }
                    }
                    upper = reduceMax(upper, upperItem);
                } else {
                    upper = op->buffer_->tensor().shape()[i];
                }
            }

            op->info_acc_lower_->emplace_back(lower);
            op->info_acc_len_->emplace_back(
                makeAdd(makeSub(upper, lower), makeIntConst(1)));
        }
    }
    return op;
}

Expr CompAccessBound::visit(const Load &op) {
    access_[op->var_].emplace_back(op->indices_); // use the old object
    return Mutator::visit(op);
}

Stmt CompAccessBound::visit(const Store &op) {
    access_[op->var_].emplace_back(op->indices_); // use the old object
    return Mutator::visit(op);
}

Stmt CompAccessBound::visit(const AddTo &op) {
    access_[op->var_].emplace_back(op->indices_); // use the old object
    return Mutator::visit(op);
}

Stmt CompAccessBound::visit(const For &op) {
    defs_.insert(op->iter_);
    auto ret = Mutator::visit(op);
    defs_.erase(op->iter_);
    return ret;
}

} // namespace ir

