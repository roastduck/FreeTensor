#include <cursor.h>

namespace ir {

Stmt Cursor::getParentById(const std::string &id) const {
    for (auto it = stack_.rbegin(); it != stack_.rend(); it++) {
        if ((*it)->id() == id) {
            return *it;
        }
    }
    return nullptr;
}

} // namespace ir

