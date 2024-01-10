#include <get_new_name.h>

namespace freetensor {

std::string getNewName(const std::string &oldName,
                       const std::unordered_set<std::string> &used) {
    for (int i = 1;; i++) {
        if (auto name = oldName + "." + std::to_string(i); !used.count(name)) {
            return name;
        }
    }
}

} // namespace freetensor
