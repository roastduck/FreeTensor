#ifndef FREE_TENSOR_VAR_REORDER_H
#define FREE_TENSOR_VAR_REORDER_H

#include <algorithm>

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

class VarReorder : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    ID def_;
    std::string var_;
    std::vector<int> order_;
    bool forceReorderInMatMul_;
    bool found_ = false;

  public:
    VarReorder(const ID &def, const std::vector<int> &order,
               bool forceReorderInMatMul)
        : def_(def), order_(order),
          forceReorderInMatMul_(forceReorderInMatMul) {
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
            for (auto &&nth : order_) {
                indices.emplace_back(op->indices_[nth]);
            }
            op->indices_ = std::move(indices);
        }
        return op;
    }

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const MatMul &op) override;
};

Stmt varReorderImpl(const Stmt &ast, const ID &def,
                    const std::vector<int> &order,
                    bool forceReorderInMatMul = false);

Stmt varReorder(const Stmt &ast, const ID &def, const std::vector<int> &order);

} // namespace freetensor

#endif // FREE_TENSOR_VAR_REORDER_H
