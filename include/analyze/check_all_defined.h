#ifndef CHECK_ALL_DEFINED_H
#define CHECK_ALL_DEFINED_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

class CheckAllDefined : public Visitor {
    const std::unordered_set<std::string> &defs_;
    bool allDef_ = true;

  public:
    CheckAllDefined(const std::unordered_set<std::string> &defs)
        : defs_(defs) {}

    bool allDef() const { return allDef_; }

  protected:
    void visit(const Var &op) override;
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

bool checkAllDefined(const std::unordered_set<std::string> &defs,
                    const AST &op);

} // namespace ir

#endif // CHECK_ALL_DEFINED_H
