#include <pass/normalize_loops.h>
#include <pass/simplify.h>

namespace freetensor {

Expr NormalizeLoops::visit(const Var &op) {
    auto &&l = loop(op->name_);
    if (filteredIn_.count(l)) {
        return makeAdd(makeMul(op, (*this)(l->step_)), (*this)(l->begin_));
    } else {
        return BaseClass::visit(op);
    }
}

Stmt NormalizeLoops::visit(const For &_op) {
    if (filter_ == nullptr || filter_(_op)) {
        filteredIn_.insert(_op);
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        filteredIn_.erase(_op);
        op->begin_ = makeIntConst(0);
        op->end_ = op->len_;
        op->step_ = makeIntConst(1);
        return op;
    } else {
        return BaseClass::visit(_op);
    }
}

Stmt normalizeLoops(const Stmt &_op,
                    const std::function<bool(const For &)> &filter) {
    auto op = NormalizeLoops{filter}(_op);
    op = simplify(op);
    return op;
}

} // namespace freetensor
