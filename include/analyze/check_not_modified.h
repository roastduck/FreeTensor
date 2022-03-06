#ifndef CHECK_NOT_MODIFIED
#define CHECK_NOT_MODIFIED

#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

enum class CheckNotModifiedSide : int { Before, After };

class CheckNameToDefMapping : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    ID pos_;
    const std::unordered_set<std::string> &names_;
    std::unordered_map<std::string, ID> name2def_;

  public:
    CheckNameToDefMapping(const ID &pos,
                          const std::unordered_set<std::string> &names)
        : pos_(pos), names_(names) {}

    const std::unordered_map<std::string, ID> &name2def() const {
        return name2def_;
    }

  protected:
    void visitStmt(const Stmt &stmt) override;
};

class InsertTmpEval : public Mutator {
    Expr expr_;
    ID s0_, s1_, s0Eval_, s1Eval_;
    CheckNotModifiedSide s0Side_, s1Side_;

  public:
    InsertTmpEval(const Expr &expr, CheckNotModifiedSide s0Side, const ID &s0,
                  CheckNotModifiedSide s1Side, const ID &s1)
        : expr_(expr), s0_(s0), s1_(s1), s0Side_(s0Side), s1Side_(s1Side) {}

    const ID &s0Eval() const { return s0Eval_; }
    const ID &s1Eval() const { return s1Eval_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
};

/**
 * Verify if `expr` is evaluated at just before/after `s0` and before/after
 * `s1`, it will result in the same value
 *
 * It will return false in two cases:
 *
 * 1. The defining VarDef or For nodes in `s0` and `s1` are different, or
 * 2. Variables used in `expr` is written between `s0` and `s1`.
 */
bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1);

} // namespace ir

#endif // CHECK_NOT_MODIFIED
