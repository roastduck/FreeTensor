#ifndef ALL_USES_H
#define ALL_USES_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Record all buffers that are used in an AST
 */
class AllUses : public Visitor {
  public:
    typedef int AllUsesType;
    static constexpr AllUsesType CHECK_LOAD = 0x1;
    static constexpr AllUsesType CHECK_STORE = 0x2;
    static constexpr AllUsesType CHECK_REDUCE = 0x4;
    static constexpr AllUsesType CHECK_VAR = 0x8;

  private:
    AllUsesType type_;
    bool noRecurseIdx_;
    std::unordered_set<std::string> uses_;

  public:
    AllUses(AllUsesType type, bool noRecurseIdx)
        : type_(type), noRecurseIdx_(noRecurseIdx) {}

    const std::unordered_set<std::string> &uses() const { return uses_; }

  protected:
    void visit(const Load &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const Var &op) override;
};

std::unordered_set<std::string>
allUses(const AST &op,
        AllUses::AllUsesType type = AllUses::CHECK_LOAD | AllUses::CHECK_STORE |
                                    AllUses::CHECK_REDUCE,
        bool noRecurseIdx = false);
std::unordered_set<std::string> allReads(const AST &op,
                                         bool noRecurseIdx = false);
std::unordered_set<std::string> allWrites(const AST &op,
                                          bool noRecurseIdx = false);
std::unordered_set<std::string> allIters(const AST &op,
                                         bool noRecurseIdx = false);
std::unordered_set<std::string> allNames(const AST &op,
                                         bool noRecurseIdx = false);

} // namespace ir

#endif // ALL_USES_H
