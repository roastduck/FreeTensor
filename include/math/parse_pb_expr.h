#ifndef FREE_TENSOR_PARSE_PB_EXPR_H
#define FREE_TENSOR_PARSE_PB_EXPR_H

#include <expr.h>

namespace freetensor {

/**
 * One contiguous factor of a PBFunc prased as ASTs
 */
struct SimplePBFuncAST {
    std::vector<std::string> args_;
    std::vector<Expr> values_;
    Expr cond_; // Maybe null
};

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

} // namespace freetensor

#endif // FREE_TENSOR_PARSE_PB_EXPR_H
