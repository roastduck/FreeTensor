#include <func.h>
namespace freetensor {

Func deepCopy(const Func &func) {
    return _makeFunc(func->name_, func->params_, func->returns_,
                     deepCopy(func->body_), func->closure_);
}

} // namespace freetensor
