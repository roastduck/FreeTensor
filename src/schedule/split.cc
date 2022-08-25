#include <pass/simplify.h>
#include <schedule/split.h>

namespace freetensor {

Stmt Splitter::visit(const For &_op) {
    if (_op->id() == src_) {
        auto iter0 = _op->iter_ + ".0";
        auto iter1 = _op->iter_ + ".1";
        auto shift = makeIntConst(shift_);
        auto shifted_len = makeAdd(_op->len_, shift);
        Expr factor, nparts;

        if (factor_ != -1) {
            ASSERT(nparts_ == -1);
            factor = makeIntConst(factor_);
            nparts = makeCeilDiv(shifted_len, factor);
        } else {
            ASSERT(nparts_ != -1);
            nparts = makeIntConst(nparts_);
            factor = makeCeilDiv(shifted_len, nparts);
        }

        auto nthIter = makeAdd(makeMul(makeVar(iter0), factor), makeVar(iter1));
        auto newIter =
            makeSub(makeAdd(_op->begin_, makeMul(nthIter, _op->step_)), shift);

        iterFrom_ = _op->iter_;
        iterTo_ = newIter;
        auto &&__op = Mutator::visit(_op);
        iterFrom_.clear();
        iterTo_ = nullptr;
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto &&op = __op.as<ForNode>();

        auto body = makeIf(
            makeLAnd(makeGE(nthIter, shift), makeLT(nthIter, shifted_len)),
            op->body_);
        auto inner =
            makeFor(iter1, makeIntConst(0), factor, makeIntConst(1), factor,
                    op->property_->withKeepSingleton(keepSingleton_), body,
                    makeMetadata("split.1", op));
        dst1_ = inner->id();
        auto outer =
            makeFor(iter0, makeIntConst(0), nparts, makeIntConst(1), nparts,
                    op->property_->withKeepSingleton(keepSingleton_), inner,
                    makeMetadata("split.0", op));
        dst0_ = outer->id();
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

std::pair<Stmt, std::pair<ID, ID>> split(const Stmt &_ast, const ID &id,
                                         int factor, int nparts, int shift,
                                         bool keepSingleton) {
    Splitter mutator(id, factor, nparts, shift, keepSingleton);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule("Loop not found");
    }
    ast = simplify(ast); // try to remove divisions, or it will hinder
                         // the dependency analysis
    return std::make_pair(ast,
                          std::make_pair(mutator.outerId(), mutator.innerId()));
}

} // namespace freetensor
