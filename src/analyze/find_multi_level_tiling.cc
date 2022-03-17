#include <analyze/find_multi_level_tiling.h>
#include <ast.h>
#include <iostream>

using std::cout;
using std::endl;

namespace ir {
void FindMultiLevelTiling::visit(const For &op) {
    if (op->len_->nodeType() != ASTNodeType::IntConst) {
        throw Error("Auto scheduling of non-constant for loop is not yet "
                    "supported.");
    }
    storeBuf();
    if (!downward && !stackMarkBranch_.empty()) {
        stackMarkBranch_.back() = true;
    }
    stack_.push_back({op->id(), op->len_.as<IntConstNode>()->val_});
    stackMarkBranch_.push_back(false);
    downward = true;
    Visitor::visit(op);
    if (stackMarkBranch_.back()) {
        storeBuf();
    }
    if (ID dest = hasStore(op); dest.isValid()) {
        storeBuf();
        buf_.push_back(stack_.back());
        bufIndices_ = forsWithStore_.at(op->id()).indices;
        bufCheckDataReuseIndices_ =
            forsWithStore_.at(op->id()).checkDataReuseIndices;
        dest_ = dest;
    } else if (!buf_.empty()) {
        buf_.push_back(stack_.back());
    }
    stack_.pop_back();
    stackMarkBranch_.pop_back();
    downward = false;
}

void FindMultiLevelTiling::storeBuf() {
    if (!buf_.empty()) {
        bool hasDataReuse = false;
        for (const auto &infoItem : bufCheckDataReuseIndices_) {
            std::vector<bool> checkAppear(buf_.size());
            for (unsigned i = 0; i < infoItem.size(); i++) {
                const auto &mapItem =
                    loopVariExprMap_.at(infoItem[i].as<ExprNode>());
                for (unsigned j = 0; j < buf_.size(); j++) {
                    if (mapItem.count(buf_[j].id) &&
                        mapItem.at(buf_[j].id) == LoopVariability::Variance) {
                        checkAppear[j] = true;
                    }
                }
            }
            for (unsigned i = 0; i < buf_.size(); i++) {
                if (!checkAppear[i]) {
                    hasDataReuse = true;
                    break;
                }
            }
            if (hasDataReuse) {
                break;
            }
        }

        if (hasDataReuse) {
            ForsWithDataReuse tmp;
            tmp.dest = dest_.strId();
            tmp.outermost = buf_[0].id;
            std::vector<bool> checkAppear(buf_.size());
            for (unsigned i = 0; i < bufIndices_.size(); i++) {
                const auto &mapItem =
                    loopVariExprMap_.at(bufIndices_[i].as<ExprNode>());
                for (unsigned j = 0; j < buf_.size(); j++) {
                    if (mapItem.count(buf_[j].id) &&
                        mapItem.at(buf_[j].id) == LoopVariability::Variance) {
                        checkAppear[j] = true;
                    }
                }
            }
            for (unsigned i = 0; i < buf_.size(); i++) {
                if (checkAppear[i]) {
                    tmp.spaceLoops.push_back(buf_[i]);
                } else {
                    tmp.reductionLoops.push_back(buf_[i]);
                }
            }
            found_.push_back(tmp);
        }

        buf_.clear();
        bufIndices_.clear();
        bufCheckDataReuseIndices_.clear();
        dest_ = "";

        // if (hasDataReuse) {
        //     const auto &nw = found_.back();
        //     std::cout << "found ";
        //     for (const auto &loop : nw.spaceLoops) {
        //         std::cout << "S " << loop.id << " ";
        //     }
        //     for (const auto &loop : nw.reductionLoops) {
        //         std::cout << "R " << loop.id << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
}

ID FindMultiLevelTiling::hasStore(const For &op) {
    if (forsWithStore_.count(op->id()))
        return forsWithStore_[op->id()].dest;
    return "";
}

void FindHasStore::visit(const For &op) {
    stack_.push_back({op->id(), op->len_.as<IntConstNode>()->val_});
    Visitor::visit(op);
    stack_.pop_back();
}

void FindHasStore::visit(const Store &op) {
    if (found_.count(stack_.back().id)) {
        ForWithStore &forWithStore = found_.at(stack_.back().id);
        forWithStore.indices.insert(forWithStore.indices.end(),
                                    op->indices_.begin(), op->indices_.end());
        forWithStore.checkDataReuseIndices.push_back(op->indices_);
    } else {
        found_.insert(
            {stack_.back().id,
             {stack_.back().id, op->id(), op->indices_,
              std::vector<std::vector<SubTree<ExprNode>>>(1, op->indices_)}});
    }
    Visitor::visit(op);
}

void FindHasStore::visit(const Load &op) {
    if (found_.count(stack_.back().id)) {
        ForWithStore &forWithStore = found_.at(stack_.back().id);
        forWithStore.checkDataReuseIndices.push_back(op->indices_);
    } else {
        throw Error(
            "A load node appearing without a store node is not supported yet.");
    }
}
} // namespace ir