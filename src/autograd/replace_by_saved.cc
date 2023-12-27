#include <autograd/replace_by_saved.h>
#include <hash.h>

namespace freetensor {

Expr ReplaceBySaved::visitExpr(const Expr &expr) {
    if (isGrad_ && invertFromStore_.has_value() &&
        invertFromStore_->match(expr)) {
        auto &&alreadyStored = invertFromStore_->store();
        std::string var = alreadyStored->var_;
        std::vector<Expr> indices = alreadyStored->indices_;
        auto dtype = symbolTable_.buffer(var)->tensor()->dtype();
        if (intermediatesMap_.count(symbolTable_.def(var)->id()) &&
            versions_.count(rootStmtID_)) {
            auto savedVar = intermediatesMap_.at(symbolTable_.def(var)->id());
            if (savedVar != var) {
                var = savedVar;
                indices.insert(indices.begin(),
                               versions_.at(alreadyStored->id()));
            }
        }
        return invertFromStore_->invert(
            makeLoad(var, std::move(indices), dtype)); // No recursion
    }
    return Mutator::visitExpr(expr);
}

Expr ReplaceBySaved::visit(const Load &_op) {
    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    auto op = __op.as<LoadNode>();
    if (intermediatesMap_.count(symbolTable_.def(_op->var_)->id()) &&
        versions_.count(StmtOrExprID(_op, rootStmtID_))) {
        auto savedVar = intermediatesMap_.at(symbolTable_.def(_op->var_)->id());
        if (savedVar != op->var_) {
            op->var_ = savedVar;
            op->indices_.insert(op->indices_.begin(),
                                versions_.at(StmtOrExprID(_op, rootStmtID_)));
        }
    }
    return op;
}

} // namespace freetensor
