#ifndef CHECK_LOOP_ORDER_H
#define CHECK_LOOP_ORDER_H

#include <string>
#include <vector>

#include <visitor.h>

namespace ir {

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

} // namespace ir

#endif // CHECK_LOOP_ORDER_H
