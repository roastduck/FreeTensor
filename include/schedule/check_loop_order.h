#ifndef FREE_TENSOR_CHECK_LOOP_ORDER_H
#define FREE_TENSOR_CHECK_LOOP_ORDER_H

#include <string>
#include <vector>

#include <visitor.h>

namespace freetensor {

/**
 * Return loops in nesting order
 */
class CheckLoopOrder : public Visitor {
    std::vector<ID> dstOrder_;
    std::vector<For> curOrder_;
    bool done_ = false;

  public:
    CheckLoopOrder(const std::vector<ID> &dstOrder) : dstOrder_(dstOrder) {}

    const std::vector<For> &order() const;

  protected:
    void visit(const For &op) override;
};

} // namespace freetensor

#endif // FREE_TENSOR_CHECK_LOOP_ORDER_H
