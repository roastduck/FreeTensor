#ifndef FREE_TENSOR_INVERT_FROM_STORE_H
#define FREE_TENSOR_INVERT_FROM_STORE_H

#include <functional>
#include <unordered_set>

#include <stmt.h>

namespace freetensor {

class InvertFromStore {
    Store store_;
    Expr yExpr_;
    std::function<Expr(const Expr &)> invertFromStore_;

  public:
    InvertFromStore(const Store &store, const Expr &yExpr,
                    const std::function<Expr(const Expr &)> &invertFromStore)
        : store_(store), yExpr_(yExpr), invertFromStore_(invertFromStore) {}

    const auto &store() const { return store_; }

    bool find(const Expr &expr) const;
    bool match(const Expr &expr) const;

    std::unordered_set<std::string>
    allReadsExcludingInversion(const Expr &expr) const;

    Expr invert(const Expr &e) const { return invertFromStore_(e); }
};

} // namespace freetensor

#endif // FREE_TENSOR_INVERT_FROM_STORE_H
