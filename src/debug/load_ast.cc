#include <antlr4-runtime.h>

#include <ast_lexer.h>
#include <ast_parser.h>
#include <debug.h>

namespace ir {

AST loadAST(const std::string &txt) {
    try {
        antlr4::ANTLRInputStream charStream(txt);
        ast_lexer lexer(&charStream);
        antlr4::CommonTokenStream tokens(&lexer);
        ast_parser parser(&tokens);
        parser.setErrorHandler(std::make_shared<antlr4::BailErrorStrategy>());
        return parser.program()->node;
    } catch (const antlr4::ParseCancellationException &e) {
        ERROR((std::string) "Parser error: " + e.what());
    }
}

} // namespace ir
