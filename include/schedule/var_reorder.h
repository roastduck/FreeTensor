#ifndef VAR_REORDER_H
#define VAR_REORDER_H

#include <algorithm>

#include <mutator.h>

namespace ir {

class VarReorder : public Mutator {
    std::string def_, var_;
    std::vector<int> order_;
    bool found_ = false;

  public:
    VarReorder(const std::string &def, const std::vector<int> &order)
        : def_(def), order_(order) {
        std::vector<int> numbers;
        numbers.reserve(order.size());
        for (int i = 0, n = order.size(); i < n; i++) {
            numbers.emplace_back(i);
        }
        if (!std::is_permutation(order.begin(), order.end(), numbers.begin())) {
            throw InvalidSchedule("The new order should be a permutation of "
                                  "the existing dimensions");
        }
    }

    bool found() const { return found_; }

  private:
    template <class T> T reorderMemAcc(const T &op) {
        if (op->var_ == var_) {
            std::vector<Expr> indices;
            indices.reserve(order_.size());
            if (order_.size() != op->indices_.size()) {
                throw InvalidSchedule("Number of dimensions in the order does "
                                      "not match the variable");
            }
            for (size_t i = 0, n = order_.size(); i < n; i++) {
                indices.emplace_back(op->indices_[order_[i]]);
            }
            op->indices_ =
                std::vector<SubTree<ExprNode>>(indices.begin(), indices.end());
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

} // namespace ir

#endif // VAR_REORDER_H
