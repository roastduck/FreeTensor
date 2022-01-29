#include <tensor.h>

namespace ir {

bool Tensor::isScalar() const {
    for (auto &&d : shape_) {
        if (d->nodeType() != ASTNodeType::IntConst) {
            return false;
        } else if (d.as<IntConstNode>()->val_ != 1) {
            return false;
        }
    }
    return true;
}

} // namespace ir

