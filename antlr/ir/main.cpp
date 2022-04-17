#include <iostream>

#include "antlr4-runtime.h"
#include "MyLexer.h"
#include "MyParser.h"
#include "MyParserBaseListener.h"

#include "ASTNode.h"

using namespace antlr4;

int main(int argc, const char* argv[]) {
    // std::ifstream stream;
    // stream.open(argv[1]);
    ANTLRFileStream input(argv[1]);
    MyLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    MyParser parser(&tokens);

    std::shared_ptr<ProgramNode> program = parser.program()->node;

    // tree::ParseTree *tree = parser.program();
    // MyParserBaseListener myListener;
    // tree::ParseTreeWalker::DEFAULT.walk(&myListener, tree);

    std::cout << parser.program() -> toStringTree() << std::endl;
    return 0;
}