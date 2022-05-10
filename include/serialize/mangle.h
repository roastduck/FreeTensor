#ifndef FREE_TENSOR_MANGLE_H
#define FREE_TENSOR_MANGLE_H

#include <string>

namespace freetensor {

std::string mangle(const std::string &name);
std::string unmangle(const std::string &name);

} // namespace freetensor

#endif // FREE_TENSOR_MANGLE_H
