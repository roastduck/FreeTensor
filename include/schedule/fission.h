#ifndef FISSION_H
#define FISSION_H

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <mutator.h>

namespace ir {

class HoistVar : public Mutator {
    std::string loop_, after_, beforeId_, afterId_;
    std::unordered_set<std::string> part0Vars_, part1Vars_;
    std::vector<VarDef> defStack_;
    std::vector<std::string> outerScopes_, innerLoops_;

    // var name -> loop id: which loops will a var cross during hoisting?
    std::unordered_map<std::string, std::vector<std::string>> xLoops_;

    bool inside_ = false, isAfter_ = false;

  public:
    HoistVar(const std::string &loop, const std::string &after)
        : loop_(loop), after_(after) {}

    const std::string &beforeId() const { return beforeId_; }
    const std::string &afterId() const { return afterId_; }
    bool found() const { return isAfter_; }

    const std::vector<std::string> &outerScopes() const { return outerScopes_; }
    const std::vector<std::string> &innerLoops() const { return innerLoops_; }

    const std::unordered_map<std::string, std::vector<std::string>> &
    xLoops() const {
        return xLoops_;
    }

  private:
    template <class T> void recordAccess(const T &op) {
        if (inside_) {
            (isAfter_ ? part1Vars_ : part0Vars_).insert(op->var_);
        }
    }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class AddDimToVar : public Mutator {
    // VarDef ID -> for ID
    std::unordered_map<std::string, std::vector<std::string>> toAdd_;
    // for ID -> For
    std::unordered_map<std::string, For> forMap_;
    // Var name -> VarDef ID
    std::unordered_map<std::string, std::string> defs_;

  public:
    AddDimToVar(
        const std::unordered_map<std::string, std::vector<std::string>> &toAdd)
        : toAdd_(toAdd) {}

  private:
    template <class T> T doAdd(T op) {
        if (toAdd_.count(defs_.at(op->var_))) {
            for (auto &&loop : toAdd_.at(defs_.at(op->var_))) {
                op->indices_.insert(op->indices_.begin(),
                                    makeVar(forMap_.at(loop)->iter_));
            }
        }
        return op;
    }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Expr visit(const Load &op) override;
};

class FissionFor : public Mutator {
    std::string loop_, after_, suffix0_, suffix1_;
    std::unordered_map<std::string, std::string> ids0_, ids1_;
    std::unordered_set<std::string> varUses_;
    bool inside_ = false, isPart0_ = true, inPart_ = false, isAfter_ = false;

  public:
    FissionFor(const std::string &loop, const std::string &after,
               const std::string &suffix0 = ".a",
               const std::string &suffix1 = ".b")
        : loop_(loop), after_(after), suffix0_(suffix0), suffix1_(suffix1) {}

    const std::unordered_map<std::string, std::string> &ids0() const {
        return ids0_;
    }
    const std::unordered_map<std::string, std::string> &ids1() const {
        return ids1_;
    }

  private:
    void markNewId(const Stmt &op, bool isPart0);

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

std::pair<Stmt, std::pair<std::unordered_map<std::string, std::string>,
                          std::unordered_map<std::string, std::string>>>
fission(const Stmt &ast, const std::string &loop, const std::string &after,
        const std::string &suffix0, const std::string &suffix1);

} // namespace ir

#endif // FISSION_H
