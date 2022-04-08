#include <sub_tree.h>

namespace ir {

size_t ASTPart::hash() {
    if (hash_ == ~0ull) {
        compHash();
    }
    return hash_;
}

void ASTPart::resetHash() {
    if (hash_ != ~0ull) {
        hash_ = ~0ull;
        if (auto p = parent(); p.isValid()) {
            p->resetHash();
        }
    }
}

} // namespace ir
