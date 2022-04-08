#include <sub_tree.h>

namespace ir {

size_t ASTPart::hash() {
    if (hash_ == ~0ull) {
        compHash();
    }
    return hash_;
}

} // namespace ir
