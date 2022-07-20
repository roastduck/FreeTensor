#ifndef FREE_TENSOR_TO_STRING_H
#define FREE_TENSOR_TO_STRING_H

#include <sstream>

namespace freetensor {

class ASTNode;
template <class T> class Ref;

// Define an `toString` function for all objects having `ostream <<` defined

template <class T>
concept HasStreamOutput = requires(std::ostream &os, T obj) {
    os << obj;
};

template <class T>
requires requires(T obj) {
    requires HasStreamOutput<T>;

    // We have a special version for AST and its subclasses
    requires !std::convertible_to<T, Ref<ASTNode>>;
}
std::string toString(const T &obj) {
    std::ostringstream os;
    os << obj;
    return os.str();
}

} // namespace freetensor

#endif // FREE_TENSOR_TO_STRING_H
