#ifndef BOUNDS_H
#define BOUNDS_H

#include <analyze/linear.h>

namespace ir {

struct Bound {
    Expr expr_;
    LinearExpr lin_;

    Bound(const Expr &expr);
    Bound(const LinearExpr &lin);
};

} // namespace ir

#endif // BOUNDS_H
