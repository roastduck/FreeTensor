#include <analyze/find_multi_level_tiling.h>
#include <ast.h>
#include <hash.h>
#include <iostream>

namespace freetensor {

void FindMultiLevelTiling::visit(const For &op) {
    if (op->len_->nodeType() != ASTNodeType::IntConst) {
        throw Error("Auto scheduling of non-constant for loop is not yet "
                    "supported.");
    }
    storeBuf();
    if (!downward && !stackMarkBranch_.empty()) {
        stackMarkBranch_.back() = true;
    }
    stack_.push_back({op->id(), -1, op->len_.as<IntConstNode>()->val_});
    stackMarkBranch_.push_back(false);
    downward = true;
    if (op->body_->nodeType() == ASTNodeType::StmtSeq) {
        auto seq = op->body_.as<StmtSeqNode>();
        if (seq->stmts_.size() == 2) {
            if (auto st = seq->stmts_[0].as<StoreNode>();
                st.isValid() && st->nodeType() == ASTNodeType::Store) {
                if (st->expr_->isConst()) {
                    nowInit_ = st;
                }
            }
        }
    }
    Visitor::visit(op);
    if (stackMarkBranch_.back()) {
        storeBuf();
    }
    if (std::string dest = hasStore(op); !dest.empty()) {
        storeBuf();
        buf_.push_back(stack_.back());
        nowFor_ = forsWithStore_.at(op->id());
    } else if (!buf_.empty()) {
        buf_.push_back(stack_.back());
    }
    stack_.pop_back();
    stackMarkBranch_.pop_back();
    downward = false;
}

void FindMultiLevelTiling::storeBuf() {
    HashComparator cmp;
    if (!buf_.empty()) {
        auto &bufCheckDataReuseIndices = nowFor_.checkDataReuseIndices;
        bool hasDataReuse = false;

        for (const auto &infoItem : bufCheckDataReuseIndices) {
            std::vector<bool> checkAppear(buf_.size());
            for (unsigned i = 0; i < infoItem.size(); i++) {
                for (unsigned j = 0; j < buf_.size(); j++) {
                    if (isVariant(loopVariExprMap_, infoItem[i], buf_[j].id)) {
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
        if (!nowFor_.readsItself) {
            hasDataReuse = false;
        }
        if (hasDataReuse) {
            ForsWithDataReuse tmp;
            tmp.dest = nowFor_.dest;
            tmp.reads = nowFor_.reads;
            tmp.outermost = buf_.rbegin()->id;
            auto &bufIndices = nowFor_.indices;
            tmp.dimIterated = std::vector<bool>(bufIndices.size(), false);
            std::vector<bool> checkAppear(buf_.size());
            for (unsigned i = 0; i < bufIndices.size(); i++) {
                for (unsigned j = 0; j < buf_.size(); j++) {
                    if (isVariant(loopVariExprMap_, bufIndices[i],
                                  buf_[j].id)) {
                        checkAppear[j] = true;
                        tmp.dimIterated[i] = true;
                        buf_[j].index = i;
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
            std::sort(tmp.spaceLoops.begin(), tmp.spaceLoops.end());
            bool valid = true;
            if (nowInit_.isValid()) {
                if (nowInit_->var_ != tmp.dest) {
                    valid = false;
                } else {
                    for (unsigned i = 0; i < bufIndices.size(); i++) {
                        if (!cmp(bufIndices[i], nowInit_->indices_[i])) {
                            valid = false;
                        }
                    }
                }
                if (valid) {
                    tmp.initStmt = nowInit_->id();
                }
            }
            if (valid) {
                found_.push_back(tmp);
            }
        }

        buf_.clear();
        nowFor_ = {};
        nowInit_ = nullptr;
    }
}

std::string FindMultiLevelTiling::hasStore(const For &op) {
    if (forsWithStore_.count(op->id()))
        return forsWithStore_[op->id()].dest;
    return "";
}

void FindHasStore::visit(const For &op) {
    stack_.push_back({op->id(), -1, op->len_.as<IntConstNode>()->val_});
    Visitor::visit(op);
    stack_.pop_back();
}

void FindHasStore::visit(const Store &op) {
    if (op->expr_->isConst()) {
        return;
    }
    if (found_.count(stack_.back().id)) {
        ForWithStore &forWithStore = found_.at(stack_.back().id);
        forWithStore.indices.insert(forWithStore.indices.end(),
                                    op->indices_.begin(), op->indices_.end());
        forWithStore.checkDataReuseIndices.push_back(op->indices_);
    } else {
        found_.insert({stack_.back().id,
                       {stack_.back().id,
                        op->var_,
                        {},
                        op->indices_,
                        std::vector<std::vector<Expr>>(1, op->indices_),
                        false}});
    }
    Visitor::visit(op);
}

void FindHasStore::visit(const Load &op) {
    if (found_.count(stack_.back().id)) {
        ForWithStore &forWithStore = found_.at(stack_.back().id);
        forWithStore.checkDataReuseIndices.push_back(op->indices_);
        if (op->var_ != forWithStore.dest) {
            forWithStore.reads.push_back(op->var_);
        } else {
            forWithStore.readsItself = true;
        }
    } else {
        throw Error(
            "A load node appearing without a store node is not supported yet.");
    }
}

} // namespace freetensor
