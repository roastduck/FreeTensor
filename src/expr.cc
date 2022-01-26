#include <expr.h>
#include <hash.h>

namespace ir {

void CommutativeBinaryExprNode::compHash() { hash_ = Hasher::compHash(*this); }
void NonCommutativeBinaryExprNode::compHash() {
    hash_ = Hasher::compHash(*this);
}
void UnaryExprNode::compHash() { hash_ = Hasher::compHash(*this); }
void AnyExprNode::compHash() { hash_ = Hasher::compHash(*this); }
void VarNode::compHash() { hash_ = Hasher::compHash(*this); }
void LoadNode::compHash() { hash_ = Hasher::compHash(*this); }
void IntConstNode::compHash() { hash_ = Hasher::compHash(*this); }
void FloatConstNode::compHash() { hash_ = Hasher::compHash(*this); }
void BoolConstNode::compHash() { hash_ = Hasher::compHash(*this); }
void IfExprNode::compHash() { hash_ = Hasher::compHash(*this); }
void CastNode::compHash() { hash_ = Hasher::compHash(*this); }
void IntrinsicNode::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace ir
