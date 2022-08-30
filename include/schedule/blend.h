#ifndef FREE_TENSOR_BLEND_H
#define FREE_TENSOR_BLEND_H

#include <unordered_map>

#include <analyze/find_loop_variance.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

class FindAllScopesInside : public Visitor {
    ID loop_;
    std::vector<ID> scopes_;
    bool inLoop_ = false;
    bool found_ = false;

  public:
    FindAllScopesInside(const ID &loop) : loop_(loop) {}

    const std::vector<ID> &scopes() const { return scopes_; }

    bool found() const { return found_; }

  protected:
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
};

class BlendPass : public Mutator {
    ID loop_;
    bool inLoop_ = false;
    std::string iter_;
    Expr begin_, step_;
    int len_ = 0, curIter_ = 0;
    std::vector<Stmt> envStack_;
    std::vector<VarDef> defs_;
    std::unordered_map<std::string, std::pair<Expr, Expr>> offset_;
    const LoopVariExprMap &exprVari_;
    const LoopVariUniqVarMap &varVari_;

  public:
    BlendPass(const ID &loop, const LoopVariExprMap &exprVari,
              const LoopVariUniqVarMap &varVari)
        : loop_(loop), exprVari_(exprVari), varVari_(varVari) {}

  private:
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
                stmt->metadata() =
                    makeMetadata("blend." + std::to_string(curIter_), stmt);

                for (auto it = envStack_.rbegin(); it != envStack_.rend();
                     it++) {
                    switch ((*it)->nodeType()) {
                    case ASTNodeType::For: {
                        auto env = it->as<ForNode>();
                        stmt = makeFor(env->iter_, (*this)(env->begin_),
                                       (*this)(env->end_), (*this)(env->step_),
                                       (*this)(env->len_), env->property_,
                                       std::move(stmt));
                        break;
                    }
                    case ASTNodeType::If: {
                        auto env = it->as<IfNode>();
                        stmt = makeIf((*this)(env->cond_), std::move(stmt));
                        break;
                    }
                    case ASTNodeType::Assert: {
                        auto env = it->as<AssertNode>();
                        stmt = makeAssert((*this)(env->cond_), std::move(stmt));
                        break;
                    }
                    default:
                        ASSERT(false);
                    }
                }
                stmts.emplace_back(std::move(stmt));
            }
            return makeStmtSeq(std::move(stmts));
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

Stmt blend(const Stmt &ast, const ID &loop);

} // namespace freetensor

#endif // FREE_TENSOR_BLEND_H
