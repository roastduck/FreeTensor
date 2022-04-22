#ifndef CHECK_ALL_DEFINED_H
#define CHECK_ALL_DEFINED_H

#include <unordered_set>

#include <ast.h>

namespace ir {

bool checkAllDefined(const std::unordered_set<std::string> &defs,
                     const AST &op);

} // namespace ir

#endif // CHECK_ALL_DEFINED_H
