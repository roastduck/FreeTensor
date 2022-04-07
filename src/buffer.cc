#include <ast.h>
#include <buffer.h>

namespace ir {

Ref<Buffer> deepCopy(const Ref<Buffer> &b) {
    return Ref<Buffer>::make(deepCopy(b->tensor()), b->atype(), b->mtype());
}

} // namespace ir
