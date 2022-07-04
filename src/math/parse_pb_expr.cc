#include <antlr4-runtime.h>

#include <debug.h>
#include <math/parse_pb_expr.h>
#include <pb_lexer.h>
#include <pb_parser.h>

namespace freetensor {

PBFuncAST parsePBFunc(const std::string &str) {
    try {
        antlr4::ANTLRInputStream charStream(str);
        pb_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        pb_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        auto &&func = parser.func();
        return func->ast;
    } catch (const antlr4::ParseCancellationException &e) {
        throw ParserError((std::string) "Parser error: " + e.what());
    }
}

SimplePBFuncAST parseSimplePBFunc(const std::string &str) {
    auto ret = parsePBFunc(str);
    if (ret.size() != 1) {
        throw ParserError(str + " is not a simple PBFunc");
    }
    return ret.front();
}

} // namespace freetensor
