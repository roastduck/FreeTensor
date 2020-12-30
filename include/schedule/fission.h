#ifndef FISSION_H
#define FISSION_H

#include <string>
#include <unordered_set>

#include <mutator.h>

namespace ir {

class HoistVar : public Mutator {
    std::string loop_, after_;
    std::unordered_set<std::string> part0Vars_, part1Vars_;
    std::vector<VarDef> defStack_;
    bool inside_ = false, isAfter_ = false;

  public:
    HoistVar(const std::string &loop, const std::string &after)
        : loop_(loop), after_(after) {}

  private:
    template <class T> void recordAccess(const T &op) {
        if (inside_) {
            (isAfter_ ? part1Vars_ : part0Vars_).insert(op->var_);
        }
    }

  protected:
    virtual Stmt visit(const For &op) override;
    virtual Stmt visit(const StmtSeq &op) override;
    virtual Stmt visit(const VarDef &op) override;
    virtual Stmt visit(const Store &op) override;
    virtual Expr visit(const Load &op) override;
    virtual Stmt visit(const AddTo &op) override;
};

class FissionFor : public Mutator {
    std::string loop_, after_;
    std::string id0_, id1_;
    std::unordered_set<std::string> varUses_;
    bool inside_ = false, isPart0_ = true, inPart_ = false;

  public:
    FissionFor(const std::string &loop, const std::string &after)
        : loop_(loop), after_(after), id0_(loop_ + ".a"), id1_(loop_ + ".b") {}

    const std::string &id0() const { return id0_; }
    const std::string &id1() const { return id1_; }

  protected:
    virtual Stmt visit(const For &op) override;
    virtual Stmt visit(const StmtSeq &op) override;
    virtual Stmt visit(const VarDef &op) override;
    virtual Stmt visit(const Store &op) override;
    virtual Expr visit(const Load &op) override;
    virtual Stmt visit(const AddTo &op) override;
};

} // namespace ir

#endif // FISSION_H
