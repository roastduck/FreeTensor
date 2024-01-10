#ifndef FREE_TENSOR_GET_NEW_NAME_H
#define FREE_TENSOR_GET_NEW_NAME_H

#include <string>
#include <unordered_set>

namespace freetensor {

std::string getNewName(const std::string &oldName,
                       const std::unordered_set<std::string> &used);

}

#endif // FREE_TENSOR_GET_NEW_NAME_H
