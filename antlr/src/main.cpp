#include <fstream>
#include <iostream>

#include "MDLexer.h"
#include "MDParser.h"
#include "ScopeMutator.h"
#include "GetTypeInfo.h"
#include "AnnotateTypeInfo.h"
#include "VarAllocVisitor.h"
#include "CodeGenVisitor.h"
#include "DataGenVisitor.h"

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <source file>" << std::endl;
        return 1;
    }

    std::ifstream stream;
    stream.open(argv[1]);
    antlr4::ANTLRInputStream input(stream);
    MDLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    MDParser parser(&tokens);

    std::shared_ptr<ProgramNode> program = parser.program()->node;

    program = ScopeMutator()(program);
    auto typeInfo = GetTypeInfo().get(program);
    program = AnnotateTypeInfo().annotate(program, typeInfo);
    auto varMap = VarAllocVisitor().allocVar(program);
    auto data = DataGenVisitor().genData(program, typeInfo);
    auto code = CodeGenVisitor().genCode(program, varMap, typeInfo);
    std::cout << data << code;

    return 0;
}

