#include <analyze/all_uses.h>
#include <analyze/check_all_defined.h>

namespace ir {

bool checkAllDefined(const std::unordered_set<std::string> &defs,
                     const AST &op) {
    for (auto &&name : allNames(op)) {
        if (!defs.count(name)) {
            return false;
        }
    }
    return true;
}

} // namespace ir
