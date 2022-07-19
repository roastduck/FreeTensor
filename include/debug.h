#ifndef FREE_TENSOR_DEBUG_H
#define FREE_TENSOR_DEBUG_H

#include <iostream>
#include <string>

// Include minimal headers

#include <ast.h>
#include <debug/logger.h>

namespace freetensor {

// Explicitly declare some functions, but not include their headers

std::string toString(const AST &op);
std::string toString(const AST &op, bool pretty);
std::string toString(const AST &op, bool pretty, bool printAllId);

bool match(const Stmt &pattern, const Stmt &instance);

std::ostream &operator<<(std::ostream &os, const AST &op);

// Define an ostream override for all objects having `toString` defined

template <class T>
concept HasToString = requires(T obj) {
    { toString(obj) } -> std::convertible_to<std::string>;
};

template <class T>
requires requires(T obj) {
    requires HasToString<T>;
    requires !std::convertible_to<T, std::string>; // No redefine if already
                                                   // implicitly convertible
    requires !std::convertible_to<T, AST>; // We have a special version for AST
                                           // and its subclasses
}
std::ostream &operator<<(std::ostream &os, const T &obj) {
    os << toString(obj);
    return os;
}

} // namespace freetensor

#endif // FREE_TENSOR_DEBUG_H
