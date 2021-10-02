#ifndef OUTPUT_INTERMEDIATES_H
#define OUTPUT_INTERMEDIATES_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>
#include <visitor.h>

namespace ir {

class CountScopeLen : public Visitor {
    std::string def_, var_;
    const std::unordered_set<std::string>
        &affectingScopes_; // For or StmtSeq IDs
    std::unordered_map<Stmt, Expr> scopeLen_;

  public:
    CountScopeLen(const std::string &def,
                  const std::unordered_set<std::string> &affectingScopes)
        : def_(def), affectingScopes_(affectingScopes) {}

    const std::unordered_map<Stmt, Expr> &scopeLen() const { return scopeLen_; }

  protected:
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const If &op) override;
    void visit(const Assert &op) override;
};

class AddExtraDim : public Mutator {
    std::string def_, var_;
    const std::unordered_set<std::string>
        &affectingScopes_; // For or StmtSeq IDs
    const std::unordered_map<Stmt, Expr> &scopeLen_;
    Expr totLen_;
    std::string tapeName_;
    Expr offset_ = makeIntConst(0);

  public:
    AddExtraDim(const std::string &def,
                const std::unordered_set<std::string> &affectingScopes,
                const std::unordered_map<Stmt, Expr> &scopeLen,
                const Expr &totLen)
        : def_(def), affectingScopes_(affectingScopes), scopeLen_(scopeLen),
          totLen_(totLen) {}

    const std::string &tapeName() const { return tapeName_; }

  private:
    template <class T> Stmt visitStoreLike(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (op->var_ == var_) {
            std::vector<Expr> newIndices(1, offset_);
            newIndices.insert(newIndices.end(), op->indices_.begin(),
                              op->indices_.end());
            auto newStore =
                makeStore("", op->var_ + ".tape", std::move(newIndices),
                          makeLoad(op->var_, op->indices_));
            return makeStmtSeq("", {op, newStore});
        }
        return op;
    }

  protected:
    Stmt visit(const Store &op) override { return visitStoreLike(op); }
    Stmt visit(const ReduceTo &op) override { return visitStoreLike(op); }
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
};

/**
 * Save some specified intermediate (MemType::Cache) variables as outputs in a
 * program
 *
 * Old intermediate variables are still preserved, but may be removed using a
 * `inline` schedule. Rationale: one intermediate (old) element maps to multiple
 * output (new) elements, so it is hard to determine which element to load
 * from, if directly loading from the output variable
 *
 * @params op : The program
 * @params intermediates : VarDef IDs of the intermediate variables
 * @return : (
 *  The transformed program
 *  Mapping from VarDef IDs of intermediate variables to output names
 * )
 */
std::pair<Stmt, std::unordered_map<std::string, std::string>>
outputIntermediates(const Stmt &op,
                    const std::unordered_set<std::string> &intermediates);

} // namespace ir

#endif // OUTPUT_INTERMEDIATES_H
