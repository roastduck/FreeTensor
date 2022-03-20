#ifndef IR_FIND_ELEMENTWISE_H
#define IR_FIND_ELEMENTWISE_H

#include <analyze/symbol_table.h>
#include <stmt.h>
#include <visitor.h>

namespace ir {

class FindSingleElementWise : public SymbolTable<Visitor> {
    Store nowStore_;
    ReduceTo nowReduceTo_;
    typedef SymbolTable<Visitor> BaseClass;
    std::string name_;
    Stmt found_;
    bool invalid_;

  public:
    explicit FindSingleElementWise(std::string name)
        : name_(std::move(name)), invalid_(false) {}
    using SymbolTable<Visitor>::visit;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Load &op) override;
    bool isElementWise(const Store &st, const Load &ld);
    bool isElementWise(const ReduceTo &st, const Load &ld);
    Stmt result() { return invalid_ ? nullptr : found_; }
};

Stmt findSingleElementWiseConsumer(const Stmt &root, const std::string &name);

} // namespace ir

#endif // IR_FIND_ELEMENTWISE_H
