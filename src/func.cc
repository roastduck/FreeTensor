#include <func.h>

namespace freetensor {

Func deepCopy(const Func &func) {
    return makeFunc(func->name_, func->params_, func->returns_,
                    deepCopy(func->body_));
}

std::ostream &operator<<(std::ostream &os, const FuncParam &p) {
    os << p.name_;
    if (p.closure_.isValid()) {
        os << " @" << p.closure_.get();
        if (p.updateClosure_) {
            os << "!";
        }
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const FuncRet &r) {
    os << r.name_ << ": " << r.dtype_;
    if (r.closure_.isValid()) {
        os << " @" << r.closure_.get();
        if (r.returnClosure_) {
            os << "!";
        }
    }
    return os;
}

} // namespace freetensor
