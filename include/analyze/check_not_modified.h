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
    Expr s0Expr_, s1Expr_;
    ID s0_, s1_, s0Eval_, s1Eval_;
    CheckNotModifiedSide s0Side_, s1Side_;

  public:
    InsertTmpEval(const Expr &s0Expr, const Expr &s1Expr,
                  CheckNotModifiedSide s0Side, const ID &s0,
                  CheckNotModifiedSide s1Side, const ID &s1)
        : s0Expr_(s0Expr), s1Expr_(s1Expr), s0_(s0), s1_(s1), s0Side_(s0Side),
          s1Side_(s1Side) {}

    const ID &s0Eval() const { return s0Eval_; }
    const ID &s1Eval() const { return s1Eval_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
};

/**
 * Verify if `expr` evaluates to the same value in the period that AFTER
 * ENCOUNTERING `s0` UNTIL ENCOUNTERING THE NEXT `s1`. The exact boundary of the
 * period to check can be set by `s0Side` and `s1Side`, to specify whether the
 * period includes `s0` and `s1` themselves
 *
 * It will return false in two cases:
 *
 * 1. Variables in `expr` is not defined at `s0` or `s1`, or
 * 2. The defining VarDef or For nodes in `s0` and `s1` are different, or
 * 3. Variables used in `expr` is written between `s0` and `s1`.
 */
bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1);

/**
 * Another version of `checkNotModified` that accpets two expressions, for `s0`
 * and `s1`, respectively
 *
 * This version of `checkNotModified` is used when the iterators in the
 * expression have different names in `s0` and `s1`
 *
 * TODO: Check the mapping of the iterators in a general way in
 * `checkNotModified`. Currently it is checked explicitly in schedule/inline and
 * pass/tensor_prop_const
 */
bool checkNotModified(const Stmt &op, const Expr &s0Expr, const Expr &s1Expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1);

} // namespace ir

#endif // CHECK_NOT_MODIFIED
