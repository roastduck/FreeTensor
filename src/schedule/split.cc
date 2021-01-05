#include <schedule/split.h>

namespace ir {

Stmt Splitter::visit(const For &_op) {
    if (_op->id() == src_) {
        auto iter0 = _op->iter_ + ".0";
        auto iter1 = _op->iter_ + ".1";
        auto len = makeSub(_op->end_, _op->begin_);
        Expr factor, nparts;

        if (factor_ != -1) {
            ASSERT(nparts_ == -1);
            factor = makeIntConst(factor_);
            nparts = makeAdd(makeDiv(makeSub(len, makeIntConst(1)), factor),
                             makeIntConst(1));
        } else {
            ASSERT(nparts_ != -1);
            nparts = makeIntConst(nparts_);
            factor = makeAdd(makeDiv(makeSub(len, makeIntConst(1)), nparts),
                             makeIntConst(1));
        }

        auto newIter =
            makeAdd(_op->begin_,
                    makeAdd(makeMul(makeVar(iter0), factor), makeVar(iter1)));

        iterFrom_ = _op->iter_;
        iterTo_ = newIter;
        auto &&__op = Mutator::visit(_op);
        iterFrom_.clear();
        iterTo_ = nullptr;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto &&op = __op.as<ForNode>();

        auto body = makeIf("", makeLT(newIter, op->end_), op->body_);
        auto inner = makeFor(dst1_, iter1, makeIntConst(0), factor, body);
        auto outer = makeFor(dst0_, iter0, makeIntConst(0), nparts, inner);
        found_ = true;
        return outer;
    } else {
        return Mutator::visit(_op);
    }
}

Expr Splitter::visit(const Var &op) {
    if (op->name_ == iterFrom_) {
        return iterTo_;
    } else {
        return Mutator::visit(op);
    }
}

} // namespace ir

