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
 * Parse a PBFunc to be ASTs, but only restricted to one contiguous factor
 */
SimplePBFuncAST parseSimplePBFunc(const std::string &str);

/**
 * Construct AST from PBSet while preserving min and max with a special hack to
 * ISL
 */
PBFuncAST parsePBFuncReconstructMinMax(const PBCtx &ctx, const PBSet &set);

} // namespace freetensor

#endif // FREE_TENSOR_PARSE_PB_EXPR_H
