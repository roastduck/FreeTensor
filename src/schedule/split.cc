#include <pass/simplify.h>
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
            nparts = makeCeilDiv(len, factor);
        } else {
            ASSERT(nparts_ != -1);
            nparts = makeIntConst(nparts_);
            factor = makeCeilDiv(len, nparts);
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
        auto inner = makeFor(dst1_, iter1, makeIntConst(0), factor, factor,
                             op->property_, body);
        auto outer = makeFor(dst0_, iter0, makeIntConst(0), nparts, nparts,
                             op->property_, inner);
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

std::pair<Stmt, std::pair<std::string, std::string>>
split(const Stmt &_ast, const std::string &id, int factor, int nparts) {
    Splitter mutator(id, factor, nparts);
    auto ast = mutator(_ast);
    if (!mutator.found()) {
        throw InvalidSchedule("Loop not found");
    }
    ast = simplifyPass(ast); // try to remove divisions, or it will hinder
                             // the dependency analysis
    return std::make_pair(ast,
                          std::make_pair(mutator.outerId(), mutator.innerId()));
}

} // namespace ir
