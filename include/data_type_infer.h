#ifndef FREE_TENSOR_TYPE_INFER_H
#define FREE_TENSOR_TYPE_INFER_H

#include <expr.h>

namespace freetensor {

/**
 * Infer the data type of a (sub)expression
 *
 * Internally called by ExprNode::dtype(). Please directly use ExprNode::dtype()
 */
class DataTypeInfer {
  public:
    static DataType infer(const VarNode &op);
    static DataType infer(const LoadNode &op);
    static DataType infer(const IntConstNode &op);
    static DataType infer(const FloatConstNode &op);
    static DataType infer(const BoolConstNode &op);
    static DataType infer(const AddNode &op);
    static DataType infer(const SubNode &op);
    static DataType infer(const MulNode &op);
    static DataType infer(const RealDivNode &op);
    static DataType infer(const FloorDivNode &op);
    static DataType infer(const CeilDivNode &op);
    static DataType infer(const RoundTowards0DivNode &op);
    static DataType infer(const ModNode &op);
    static DataType infer(const RemainderNode &op);
    static DataType infer(const MinNode &op);
    static DataType infer(const MaxNode &op);
    static DataType infer(const LTNode &op);
    static DataType infer(const LENode &op);
    static DataType infer(const GTNode &op);
    static DataType infer(const GENode &op);
    static DataType infer(const EQNode &op);
    static DataType infer(const NENode &op);
    static DataType infer(const LAndNode &op);
    static DataType infer(const LOrNode &op);
    static DataType infer(const LNotNode &op);
    static DataType infer(const SqrtNode &op);
    static DataType infer(const ExpNode &op);
    static DataType infer(const SquareNode &op);
    static DataType infer(const SigmoidNode &op);
    static DataType infer(const TanhNode &op);
    static DataType infer(const AbsNode &op);
    static DataType infer(const FloorNode &op);
    static DataType infer(const CeilNode &op);
    static DataType infer(const IfExprNode &op);
    static DataType infer(const CastNode &op);
    static DataType infer(const IntrinsicNode &op);
};

} // namespace freetensor

#endif // FREE_TENSOR_TYPE_INFER_H
