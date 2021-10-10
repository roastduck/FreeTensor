#include <func.h>
namespace ir {

Func deepCopy(const Func &func) {
    return _makeFunc(func->name_, func->params_, deepCopy(func->body_),
                     func->src_);
}

} // namespace ir
