#ifndef FREE_TENSOR_AST_PARSER_BASE_H
#define FREE_TENSOR_AST_PARSER_BASE_H

#include <string>
#include <unordered_map>

#include <antlr4-runtime.h>

#include <type/data_type.h>

namespace freetensor {

class ASTParserBase : public antlr4::Parser {
  protected:
    std::unordered_map<std::string, DataType> name2dtype_;

    ASTParserBase(antlr4::TokenStream *input) : antlr4::Parser(input) {}
};

} // namespace freetensor

#endif // FREE_TENSOR_AST_PARSER_BASE_H
