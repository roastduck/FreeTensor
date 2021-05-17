#ifndef CHECK_NOT_MODIFIED
#define CHECK_NOT_MODIFIED

#include <mutator.h>

namespace ir {

class InsertTmpEval : public Mutator {
    Expr expr_;
    std::string s0_, s1_, s0Eval_, s1Eval_;

  public:
    InsertTmpEval(const Expr &expr, const std::string &s0,
                  const std::string &s1)
        : expr_(expr), s0_(s0), s1_(s1) {}

    const std::string &s0Eval() const { return s0Eval_; }
    const std::string &s1Eval() const { return s1Eval_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
};

/**
 * Verify if `expr` is evaluated at just after `s0` and before `s1`, it will
 * result in the same value
 */
bool checkNotModified(const Stmt &op, const Expr &expr, const std::string &s0,
                      const std::string &s1);

} // namespace ir

#endif // CHECK_NOT_MODIFIED
