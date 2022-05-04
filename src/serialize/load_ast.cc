#include <antlr4-runtime.h>

#include <ast_lexer.h>
#include <ast_parser.h>
#include <serialize/load_ast.h>

namespace freetensor {

AST loadAST(const std::string &txt) {
    try {
        antlr4::ANTLRInputStream charStream(txt);
        ast_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        ast_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        return parser.program()->node;
    } catch (const antlr4::ParseCancellationException &e) {
        throw ParserError((std::string) "Parser error: " + e.what());
    }
}

} // namespace freetensor
