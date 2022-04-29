#include <antlr4-runtime.h>

#include <debug.h>
#include <math/parse_pb_expr.h>
#include <pb_lexer.h>
#include <pb_parser.h>

namespace freetensor {

std::tuple<std::vector<std::string>, std::vector<Expr>, Expr>
parsePBFunc(const std::string &str) {
    try {
        antlr4::ANTLRInputStream charStream(str);
        pb_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        pb_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        auto &&func = parser.func();
        return std::make_tuple(func->args, func->values, func->cond);
    } catch (const antlr4::ParseCancellationException &e) {
        ERROR((std::string) "Parser error: " + e.what());
    }
}

} // namespace freetensor
