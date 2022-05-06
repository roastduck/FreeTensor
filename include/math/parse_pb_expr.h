#ifndef FREE_TENSOR_PARSE_PB_EXPR_H
#define FREE_TENSOR_PARSE_PB_EXPR_H

#include <expr.h>

namespace freetensor {

std::tuple<std::vector<std::string>, std::vector<Expr>, Expr>
parsePBFunc(const std::string &str);

}

#endif // FREE_TENSOR_PARSE_PB_EXPR_H
