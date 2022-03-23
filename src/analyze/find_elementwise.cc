#include <analyze/find_elementwise.h>
#include <hash.h>
namespace ir {

void FindSingleElementWise::visit(const Store &op) {
    if (invalid_) {
        return;
    }
    nowStore_ = op;
    BaseClass::visit(op);
    nowStore_ = nullptr;
}

void FindSingleElementWise::visit(const ReduceTo &op) {
    if (invalid_) {
        return;
    }
    nowReduceTo_ = op;
    BaseClass::visit(op);
    nowReduceTo_ = nullptr;
}

bool FindSingleElementWise::isElementWise(const Store &st, const Load &ld) {
    const auto &destShape = buffer(st->var_)->tensor().shape();
    const auto &srcShape = buffer(ld->var_)->tensor().shape();
    if (destShape.size() != srcShape.size()) {
        return false;
    }
    HashComparator comp;
    for (size_t i = 0; i < destShape.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(destShape[i], srcShape[i])) {
                return false;
            }
        } else if (!comp(destShape[i], srcShape[i]) &&
                   !comp(srcShape[i], makeIntConst(1))) {
            return false;
        }
    }
    for (size_t i = 0; i < st->indices_.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(st->indices_[i], ld->indices_[i])) {
                return false;
            }
        }
    }
    return true;
}

bool FindSingleElementWise::isElementWise(const ReduceTo &st, const Load &ld) {
    const auto &destShape = buffer(st->var_)->tensor().shape();
    const auto &srcShape = buffer(ld->var_)->tensor().shape();
    if (destShape.size() != srcShape.size()) {
        return false;
    }
    HashComparator comp;
    for (size_t i = 0; i < destShape.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(destShape[i], srcShape[i])) {
                return false;
            }
        } else if (!comp(destShape[i], srcShape[i]) &&
                   !comp(srcShape[i], makeIntConst(1))) {
            return false;
        }
    }
    for (size_t i = 0; i < st->indices_.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(st->indices_[i], ld->indices_[i])) {
                return false;
            }
        }
    }
    return true;
}

void FindSingleElementWise::visit(const Load &op) {
    if (invalid_) {
        return;
    }
    if (nowStore_.isValid() && op->var_ != nowStore_->var_ &&
        op->var_ == fors_.dest) {
        if (!found_.isValid() && isElementWise(nowStore_, op)) {
            found_ = nowStore_;
        } else {
            invalid_ = true;
            return;
        }
    } else if (nowReduceTo_.isValid() && op->var_ != nowReduceTo_->var_ &&
               op->var_ == fors_.dest) {
        if (!found_.isValid() && isElementWise(nowReduceTo_, op)) {
            found_ = nowStore_;
        } else {
            invalid_ = true;
            return;
        }
    }
    BaseClass::visit(op);
}

Stmt findSingleElementWiseConsumer(const Stmt &root,
                                   const ForsWithDataReuse &fors) {
    FindSingleElementWise finder(fors);
    finder(root);
    return finder.result();
}

} // namespace ir
