#include <sub_tree.h>

namespace freetensor {

int ASTPart::depth() const {
    int depth = 0;
    for (auto p = parent(); p.isValid(); p = p->parent()) {
        depth++;
    }
    return depth;
}

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

Ref<ASTPart> lca(const Ref<ASTPart> &lhs, const Ref<ASTPart> &rhs) {
    auto l = lhs, r = rhs;
    auto dl = l->depth(), dr = r->depth();
    while (dl > dr) {
        l = l->parent(), dl--;
    }
    while (dr > dl) {
        r = r->parent(), dr--;
    }
    while (l.isValid() && l != r) {
        l = l->parent();
        r = r->parent();
    }
    return l;
}

} // namespace freetensor
