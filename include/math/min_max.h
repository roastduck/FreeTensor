#ifndef FREE_TENSOR_MIN_MAX_H
#define FREE_TENSOR_MIN_MAX_H

#include <functional>

#include <expr.h>

namespace freetensor {

Expr makeMinMaxImpl(const std::vector<std::vector<Expr>> &exprs,
                    const std::function<Expr()> &negInf,
                    const std::function<Expr()> &inf);
Expr makeMaxMinImpl(const std::vector<std::vector<Expr>> &exprs,
                    const std::function<Expr()> &negInf,
                    const std::function<Expr()> &inf);

inline std::function<Expr()> asExprGenerator(std::nullptr_t) {
    return []() { return nullptr; };
}
inline std::function<Expr()> asExprGenerator(const std::function<Expr()> &e) {
    return e;
}
inline std::function<Expr()> asExprGenerator(const Expr &e) {
    return [&e]() { return e; };
}

/**
 * Make min(max(...), max(...), ...) and remove duplications
 *
 * @param exprs : Vector of vector or exprs. Items of inner ones are combined
 * with max. Items of the outer one are combined with min.
 * @param negInf : Expr or function returning an Expr. What to return in case of
 * any `max` term is empty. Can be nullptr.
 * @param inf : Expr or function returning an Expr. What to return in case of
 * the `min` term is empty. Can be nullptr.
 */
template <typename T, typename U>
Expr makeMinMax(const std::vector<std::vector<Expr>> &exprs, const T &negInf,
                const U &inf) {
    return makeMinMaxImpl(exprs, asExprGenerator(negInf), asExprGenerator(inf));
}

/**
 * Make max(min(...), min(...), ...) and remove duplications
 *
 * @param exprs : Vector of vector or exprs. Items of inner ones are combined
 * with min. Items of the outer one are combined with max.
 * @param inf : Expr or function returning an Expr. What to return in case of
 * any `in` term is empty. Can be nullptr.
 * @param negInf : Expr or function returning an Expr. What to return in case of
 * the `max` term is empty. Can be nullptr.
 */
template <typename T, typename U>
Expr makeMaxMin(const std::vector<std::vector<Expr>> &exprs, const T &inf,
                const U &negInf) {
    return makeMaxMinImpl(exprs, asExprGenerator(inf), asExprGenerator(negInf));
}

/**
 * Make l_or(l_and(...), l_and(...), ...) and remove duplications
 *
 * This will remove some duplicated items in advance, to reduce the burden on
 * simplifier
 *
 * Always returning non-null
 */
Expr makeLOrLAnd(const std::vector<std::vector<Expr>> &exprs);

} // namespace freetensor

#endif // FREE_TENSOR_MIN_MAX_H
