#ifndef IR_FIND_ELEMENTWISE_H
#define IR_FIND_ELEMENTWISE_H

#include <analyze/find_multi_level_tiling.h>
#include <analyze/symbol_table.h>
#include <stack>
#include <stmt.h>
#include <visitor.h>

namespace ir {

class FindSingleElementWise : public SymbolTable<Visitor> {
    Store nowStore_;
    ReduceTo nowReduceTo_;
    typedef SymbolTable<Visitor> BaseClass;
    ForsWithDataReuse fors_;
    ElementWiseInfo found_;
    bool invalid_;
    std::vector<std::pair<ForInfo, std::string>> stack_;

  public:
    explicit FindSingleElementWise(ForsWithDataReuse fors)
        : fors_(std::move(fors)), invalid_(false) {}
    using SymbolTable<Visitor>::visit;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Load &op) override;
    void visit(const For &op) override;
    ElementWiseInfo isElementWise(const Store &st, const Load &ld);
    ElementWiseInfo isElementWise(const ReduceTo &st, const Load &ld);
    ElementWiseInfo result() { return invalid_ ? ElementWiseInfo() : found_; }
};

ElementWiseInfo findSingleElementWiseConsumer(const Stmt &root,
                                              const ForsWithDataReuse &fors);

} // namespace ir

#endif // IR_FIND_ELEMENTWISE_H
