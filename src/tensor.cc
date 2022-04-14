#include <hash.h>
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

void Tensor::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace ir
