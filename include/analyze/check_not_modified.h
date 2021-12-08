#ifndef CHECK_NOT_MODIFIED
#define CHECK_NOT_MODIFIED

#include <mutator.h>

namespace ir {

enum class CheckNotModifiedSide : int { Before, After };

class InsertTmpEval : public Mutator {
    Expr expr_;
    std::string s0_, s1_, s0Eval_, s1Eval_;
    CheckNotModifiedSide s0Side_, s1Side_;

  public:
    InsertTmpEval(const Expr &expr, CheckNotModifiedSide s0Side,
                  const std::string &s0, CheckNotModifiedSide s1Side,
                  const std::string &s1)
        : expr_(expr), s0_(s0), s1_(s1), s0Side_(s0Side), s1Side_(s1Side) {}

    const std::string &s0Eval() const { return s0Eval_; }
    const std::string &s1Eval() const { return s1Eval_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
};

/**
 * Verify if `expr` is evaluated at just before/after `s0` and before/after
 * `s1`, it will result in the same value
 */
bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const std::string &s0,
                      CheckNotModifiedSide s1Side, const std::string &s1);

} // namespace ir

#endif // CHECK_NOT_MODIFIED
