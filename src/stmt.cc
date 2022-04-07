#include <hash.h>
#include <stmt.h>

namespace ir {

void AnyNode::compHash() { hash_ = Hasher::compHash(*this); }
void StmtSeqNode::compHash() { hash_ = Hasher::compHash(*this); }
void VarDefNode::compHash() { hash_ = Hasher::compHash(*this); }
void StoreNode::compHash() { hash_ = Hasher::compHash(*this); }
void ReduceToNode::compHash() { hash_ = Hasher::compHash(*this); }
void ForNode::compHash() { hash_ = Hasher::compHash(*this); }
void IfNode::compHash() { hash_ = Hasher::compHash(*this); }
void AssertNode::compHash() { hash_ = Hasher::compHash(*this); }
void AssumeNode::compHash() { hash_ = Hasher::compHash(*this); }
void EvalNode::compHash() { hash_ = Hasher::compHash(*this); }
void MatMulNode::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace ir
