#include <deque>

#include "ScopeMutator.h"

std::shared_ptr<StmtNode> FlattenMutator::mutate(const StmtSeqNode *_op) {
    auto op = AS(Mutator::mutate(_op), StmtSeq);
    std::vector<std::shared_ptr<StmtNode>> stmts;
    stmts.reserve(op->stmts_.size());
    for (auto &&item : op->stmts_) {
        if (item->nodeType() != ASTNodeType::StmtSeq) {
            stmts.push_back(item);
        } else {
            const StmtSeqNode *sub = static_cast<const StmtSeqNode*>(item.get());
            if (sub->isBlock_) {
                stmts.push_back(item);
            } else {
                for (auto &&subitem : sub->stmts_) {
                    stmts.push_back(subitem);
                }
            }
        }
    }
    return StmtSeqNode::make(stmts, op->isBlock_);
}

std::shared_ptr<StmtNode> VarScopeMutator::mutate(const StmtSeqNode *_op) {
    typedef std::deque<std::shared_ptr<StmtNode>> Deq;
    typedef std::vector<std::shared_ptr<StmtNode>> Vec;

    auto op = AS(Mutator::mutate(_op), StmtSeq);
    Deq stmts;
    for (auto n = op->stmts_.size(), i = n - 1; ~i; i--) {
        if (i != n - 1 && op->stmts_[i + 1]->nodeType() == ASTNodeType::VarDef &&
                op->stmts_[i]->nodeType() != ASTNodeType::VarDef) {
            stmts = Deq({StmtSeqNode::make(Vec(stmts.begin(), stmts.end()), true)});
        }
        stmts.push_front(op->stmts_[i]);
    }
    return StmtSeqNode::make(Vec(stmts.begin(), stmts.end()), op->isBlock_);
}

