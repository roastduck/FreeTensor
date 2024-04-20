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
 *
 * @{
 */
PBFuncAST parsePBFunc(const PBFunc::Serialized &f);
PBFuncAST parsePBFunc(const PBSingleFunc::Serialized &f);
/** @} */

/**
 * Construct AST from PBSet while preserving min and max with a special hack to
 * ISL
 *
 * @{
 */
PBFuncAST parsePBFuncReconstructMinMax(const PBSet &set);
PBFuncAST parsePBFuncReconstructMinMax(const PBMap &map);
/** @} */

/**
 * Parse a PBFunc to be ASTs, but only restricted to one contiguous factor
 *
 * @{
 */
inline SimplePBFuncAST parseSimplePBFunc(const auto &f) {
    auto ret = parsePBFunc(f);
    if (ret.size() != 1) {
        throw ParserError(FT_MSG << f << " is not a simple PBFunc");
    }
    return ret.front();
}
inline SimplePBFuncAST parseSimplePBFuncReconstructMinMax(const auto &f) {
    auto ret = parsePBFuncReconstructMinMax(f);
    if (ret.size() != 1) {
        throw ParserError(FT_MSG << f << " is not a simple PBFunc");
    }
    return ret.front();
}
/** @} */

} // namespace freetensor

#endif // FREE_TENSOR_PARSE_PB_EXPR_H
