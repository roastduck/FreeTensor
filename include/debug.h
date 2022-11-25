#ifndef FREE_TENSOR_DEBUG_H
#define FREE_TENSOR_DEBUG_H

#include <string>

// Include minimal headers

#include <ast.h>
#include <debug/logger.h>

namespace freetensor {

// Explicitly declare some functions, but not include their headers

std::string toString(const AST &op);
std::string toString(const AST &op, bool pretty);
std::string toString(const AST &op, bool pretty, bool printAllId);

std::ostream &operator<<(std::ostream &os, const AST &op);

bool match(const Stmt &pattern, const Stmt &instance);

void checkConflictId(const Stmt &ast);

} // namespace freetensor

#endif // FREE_TENSOR_DEBUG_H
