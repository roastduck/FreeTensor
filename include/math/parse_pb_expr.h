#ifndef FREE_TENSOR_PARSE_PB_EXPR_H
#define FREE_TENSOR_PARSE_PB_EXPR_H

#include <iostream>

#include <expr.h>
#include <math/presburger.h>

namespace freetensor {

/**
 * One contiguous factor of a PBFunc prased as ASTs
 */
struct SimplePBFuncAST {
    std::vector<std::string> args_;
    std::vector<Expr> values_;
    Expr cond_; // Maybe null
};

std::ostream &operator<<(std::ostream &os, const SimplePBFuncAST &ast);

/**
 * A PBFunc parsed as ASTs
 */
typedef std::vector<SimplePBFuncAST> PBFuncAST;

/**
 * Parse a PBFunc to be ASTs
 */
PBFuncAST parsePBFunc(const std::string &str);

/**
 * Construct AST from PBSet while preserving min and max with a special hack to
 * ISL
 *
 * @{
 */
PBFuncAST parsePBFuncReconstructMinMax(const PBCtx &ctx, const PBSet &set);
PBFuncAST parsePBFuncReconstructMinMax(const PBCtx &ctx, const PBMap &map);
/** @} */

/**
 * Parse a PBFunc to be ASTs, but only restricted to one contiguous factor
 *
 * @{
 */
inline SimplePBFuncAST parseSimplePBFunc(const std::string &str) {
    auto ret = parsePBFunc(str);
    if (ret.size() != 1) {
        throw ParserError(str + " is not a simple PBFunc");
    }
    return ret.front();
}
inline SimplePBFuncAST parseSimplePBFuncReconstructMinMax(const PBCtx &ctx,
                                                          const auto &f) {
    auto ret = parsePBFuncReconstructMinMax(ctx, f);
    if (ret.size() != 1) {
        throw ParserError(FT_MSG << f << " is not a simple PBFunc");
    }
    return ret.front();
}
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_PARSE_PB_EXPR_H
