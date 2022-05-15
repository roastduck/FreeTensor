#include <analyze/find_elementwise.h>
#include <hash.h>
namespace freetensor {

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

ElementWiseInfo FindSingleElementWise::isElementWise(const Store &st,
                                                     const Load &ld) {
    std::cout << "elementwise: " << toString(st) << "\n " << toString(ld) << std::endl;
    const auto &destShape = buffer(st->var_)->tensor()->shape();
    const auto &srcShape = buffer(ld->var_)->tensor()->shape();
    if (destShape.size() != srcShape.size()) {
        return {};
    }
    HashComparator comp;
    size_t iteratedDims = 0;
    for (size_t i = 0; i < destShape.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(destShape[i], srcShape[i])) {
                return {};
            }
            iteratedDims++;
        } else if (!comp(destShape[i], srcShape[i]) &&
                   !comp(srcShape[i], makeIntConst(1))) {
            return {};
        }
    }
    std::vector<ForInfo> forInfos;
    std::vector<size_t> indices;
    for (size_t i = 0; i < st->indices_.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(st->indices_[i], ld->indices_[i])) {
                return {};
            }
            for (size_t j = stack_.size() - 1; j >= 0; j--) {
                if (comp(makeVar(stack_[j].second), st->indices_[i])) {
                    stack_[j].first.index = i;
                    forInfos.push_back(stack_[j].first);
                    indices.push_back(j);
                    break;
                }
            }
        }
    }
    if (indices.size() != iteratedDims) {
        return {};
    }
    std::sort(indices.begin(), indices.end());
    if (indices.back() != stack_.size() - 1) {
        return {};
    }
    for (size_t i = 1; i < indices.size(); i++) {
        if (indices[i] != indices[i - 1] + 1) {
            return {};
        }
    }
    std::sort(forInfos.begin(), forInfos.end());
    for (size_t i = 0; i < forInfos.size(); i++) {
        if (forInfos[i].index != fors_.spaceLoops[i].index ||
            forInfos[i].length != fors_.spaceLoops[i].length) {
            return {};
        }
    }
    return {forInfos};
}

ElementWiseInfo FindSingleElementWise::isElementWise(const ReduceTo &st,
                                                     const Load &ld) {
    const auto &destShape = buffer(st->var_)->tensor()->shape();
    const auto &srcShape = buffer(ld->var_)->tensor()->shape();
    if (destShape.size() != srcShape.size()) {
        return {};
    }
    HashComparator comp;
    size_t iteratedDims = 0;
    for (size_t i = 0; i < destShape.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(destShape[i], srcShape[i])) {
                return {};
            }
            iteratedDims++;
        } else if (!comp(destShape[i], srcShape[i]) &&
                   !comp(srcShape[i], makeIntConst(1))) {
            return {};
        }
    }
    std::vector<ForInfo> forInfos;
    std::vector<size_t> indices;
    for (size_t i = 0; i < st->indices_.size(); i++) {
        if (fors_.dimIterated[i]) {
            if (!comp(st->indices_[i], ld->indices_[i])) {
                return {};
            }
            for (size_t j = stack_.size() - 1; j >= 0; j--) {
                if (comp(makeVar(stack_[j].second), st->indices_[i])) {
                    stack_[j].first.index = i;
                    forInfos.push_back(stack_[j].first);
                    indices.push_back(j);
                    break;
                }
            }
        }
    }
    if (indices.size() != iteratedDims) {
        return {};
    }
    std::sort(indices.begin(), indices.end());
    if (indices.back() != stack_.size() - 1) {
        return {};
    }
    for (size_t i = 1; i < indices.size(); i++) {
        if (indices[i] != indices[i - 1] + 1) {
            return {};
        }
    }
    std::sort(forInfos.begin(), forInfos.end());
    for (size_t i = 0; i < forInfos.size(); i++) {
        if (forInfos[i].index != fors_.spaceLoops[i].index ||
            forInfos[i].length != fors_.spaceLoops[i].length) {
            return {};
        }
    }
    return {forInfos};
}

void FindSingleElementWise::visit(const Load &op) {
    if (invalid_) {
        return;
    }
    if (nowStore_.isValid() && op->var_ != nowStore_->var_ &&
        op->var_ == fors_.dest) {
        if (auto info = isElementWise(nowStore_, op); info.isValid()) {
            if (!found_.isValid()) {
                found_ = info;
            } else {
                invalid_ = true;
                return;
            }
        }
    } else if (nowReduceTo_.isValid() && op->var_ != nowReduceTo_->var_ &&
               op->var_ == fors_.dest) {
        if (auto info = isElementWise(nowReduceTo_, op); info.isValid()) {
            if (!found_.isValid()) {
                found_ = info;
            } else {
                invalid_ = true;
                return;
            }
        }
    }
    BaseClass::visit(op);
}

void FindSingleElementWise::visit(const For &op) {
    stack_.emplace_back(
        ForInfo{op->id(), -1, op->len_.as<IntConstNode>()->val_}, op->iter_);
    BaseClass::visit(op);
}

ElementWiseInfo findSingleElementWiseConsumer(const Stmt &root,
                                              const ForsWithDataReuse &fors) {
    FindSingleElementWise finder(fors);
    finder(root);
    return finder.result();
}

} // namespace freetensor
