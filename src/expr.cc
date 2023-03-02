#include <data_type_infer.h>
#include <expr.h>
#include <hash.h>

namespace freetensor {

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
void LoadAtVersionNode::compHash() { hash_ = Hasher::compHash(*this); }

void VarNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LoadNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void IntConstNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void FloatConstNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void BoolConstNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void AddNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void SubNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void MulNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void RealDivNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void FloorDivNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void CeilDivNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void RoundTowards0DivNode::inferDType() {
    dtype_ = DataTypeInfer::infer(*this);
}
void ModNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void RemainderNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void MinNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void MaxNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LTNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LENode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void GTNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void GENode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void EQNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void NENode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LAndNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LOrNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LNotNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void SqrtNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void ExpNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void SquareNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void SigmoidNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void TanhNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void AbsNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void FloorNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void CeilNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void IfExprNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void CastNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void IntrinsicNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }
void LoadAtVersionNode::inferDType() { dtype_ = DataTypeInfer::infer(*this); }

} // namespace freetensor
