#ifndef BLEND_H
#define BLEND_H

#include <unordered_set>

#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllScopesInside : public Visitor {
    std::string loop_;
    std::vector<std::string> scopes_;
    bool inLoop_ = false;
    bool found_ = false;

  public:
    FindAllScopesInside(const std::string &loop) : loop_(loop) {}

    const std::vector<std::string> &scopes() const { return scopes_; }

    bool found() const { return found_; }

  protected:
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

class BlendPass : public Mutator {
    struct Env {
        Stmt env_;
        bool isVari_;
        Env(const Stmt &env, bool isVari) : env_(env), isVari_(isVari) {}
    };

    std::string loop_;
    bool inLoop_ = false;
    std::string iter_;
    Expr begin_;
    int len_ = 0, curIter_ = 0;
    std::vector<Env> envStack_;
    std::vector<VarDef> defs_;
    const std::unordered_map<Expr, std::unordered_set<std::string>> &loopVari_;

  public:
    BlendPass(const std::string &loop,
              const std::unordered_map<Expr, std::unordered_set<std::string>>
                  &loopVari)
        : loop_(loop), loopVari_(loopVari) {}

  private:
    bool checkVari(const Expr &expr) const;

    template <class T> Stmt visitLeafStmt(const T &op) {
        if (inLoop_) {
            std::vector<Stmt> stmts;
            for (curIter_ = 0; curIter_ < len_; curIter_++) {
                auto stmt = Mutator::visit(op);
                if (stmt->nodeType() == ASTNodeType::Store) {
                    stmt = visitMemAccess(stmt.template as<StoreNode>());
                } else if (stmt->nodeType() == ASTNodeType::ReduceTo) {
                    stmt = visitMemAccess(stmt.template as<ReduceToNode>());
                }

                for (auto it = envStack_.rbegin(); it != envStack_.rend();
                     it++) {
                    if (!it->isVari_) {
                        break;
                    }
                    switch (it->env_->nodeType()) {
                    case ASTNodeType::For: {
                        auto env = it->env_.as<ForNode>();
                        stmt = makeFor("", env->iter_, (*this)(env->begin_),
                                       (*this)(env->end_), env->parallel_,
                                       env->unroll_, std::move(stmt));
                        break;
                    }
                    case ASTNodeType::If: {
                        auto env = it->env_.as<IfNode>();
                        stmt = makeIf("", (*this)(env->cond_), std::move(stmt));
                        break;
                    }
                    case ASTNodeType::Assert: {
                        auto env = it->env_.as<AssertNode>();
                        stmt = makeAssert("", (*this)(env->cond_),
                                          std::move(stmt));
                        break;
                    }
                    default:
                        ASSERT(false);
                    }
                }
                stmts.emplace_back(std::move(stmt));
            }
            return makeStmtSeq("", std::move(stmts));
        } else {
            return Mutator::visit(op);
        }
    }

    template <class T> T visitMemAccess(const T &op) {
        if (inLoop_) {
            for (auto &&def : defs_) {
                if (def->name_ == op->var_) {
                    op->var_ += "." + std::to_string(curIter_);
                }
            }
        }
        return op;
    }

  protected:
    Stmt visit(const Store &op) override { return visitLeafStmt(op); }
    Stmt visit(const ReduceTo &op) override { return visitLeafStmt(op); }
    Stmt visit(const Eval &op) override { return visitLeafStmt(op); }
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Var &op) override;
    Expr visit(const Load &op) override;
};

} // namespace ir

#endif // BLEND_H
