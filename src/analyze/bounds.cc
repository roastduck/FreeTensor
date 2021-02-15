#include <algorithm>
#include <climits>
#include <functional>

#include <analyze/bounds.h>

namespace ir {

Bound::Bound(const Expr &expr)
    : expr_(expr), lin_{{{getHash(expr), {1, expr}}}, 0} {}

Bound::Bound(const LinearExpr &lin) : lin_(lin) {
    Expr b = makeIntConst(lin.bias_);
    for (auto &&item : lin.coeff_) {
        int k = item.second.k;
        auto &&a = item.second.a;

        if (k == 0) {
            continue;
        }
        Expr x;
        if (a->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(k * a.as<IntConstNode>()->val_);
        } else if (k == 1) {
            x = a;
        } else {
            x = makeMul(makeIntConst(k), a);
        }
        if (x->nodeType() == ASTNodeType::IntConst &&
            b->nodeType() == ASTNodeType::IntConst) {
            x = makeIntConst(x.as<IntConstNode>()->val_ +
                             b.as<IntConstNode>()->val_);
        } else if (b->nodeType() == ASTNodeType::IntConst &&
                   b.as<IntConstNode>()->val_ == 0) {
            // do nothing
        } else {
            x = makeAdd(x, b);
        }

        b = std::move(x);
    }
    expr_ = std::move(b);
}

} // namespace ir
