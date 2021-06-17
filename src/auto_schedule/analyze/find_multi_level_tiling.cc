//
// Created by hitonami on 2021/4/18.
//

#include <auto_schedule/analyze/find_multi_level_tiling.h>
#include <iostream>
using std::cout;
using std::endl;
namespace ir {
void FindMultiLevelTiling::visit(const For &op) {
    innermost_ = true;
    cout << "visiting " << op->id() << endl;
    if (op->begin_->nodeType() != ASTNodeType::IntConst || op->end_->nodeType() != ASTNodeType::IntConst) {
        throw Error("Auto scheduling of non-constant for loop is not yet supported.");
    }
    stack_.push_back({op->id(), op->begin_.as<IntConstNode>()->val_, op->end_.as<IntConstNode>()->val_});
    Visitor::visit(op);
    if (innermost_) {
        if (stack_.size() >= 3) {
            auto sz = stack_.size();
            found_.push_back({stack_[sz - 3], stack_[sz - 2], stack_[sz - 1]});
            const auto &nw = found_.back();
            std::cout << "found " << nw.i.id << " " << nw.j.id << " " << nw.k.id << std::endl;
        }
    }
    innermost_ = false;
    stack_.pop_back();
}
}