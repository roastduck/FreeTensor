#ifndef FREE_TENSOR_FISSION_H
#define FREE_TENSOR_FISSION_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

enum class FissionSide : int { Before, After };
inline std::ostream &operator<<(std::ostream &os, FissionSide side) {
    return os << (side == FissionSide::Before ? "before" : "after");
}

class HoistVar : public Mutator {
    ID loop_;
    ID before_, after_;
    std::vector<std::pair<ID, ID>> scopePairs_;
    std::unordered_set<std::string> part0Vars_, part1Vars_;
    std::vector<VarDef> defStack_;
    std::vector<ID> outerScopes_, innerLoops_;

    // var name -> loop id: which loops will a var cross during hoisting?
    std::unordered_map<std::string, std::vector<ID>> xLoops_;

    bool inside_ = false, isAfter_ = false;

  public:
    HoistVar(const ID &loop, const ID &before, const ID &after)
        : loop_(loop), before_(before), after_(after) {}

    const std::vector<std::pair<ID, ID>> &scopePairs() const {
        return scopePairs_;
    }

    bool found() const { return isAfter_; }

    const std::vector<ID> &outerScopes() const { return outerScopes_; }
    const std::vector<ID> &innerLoops() const { return innerLoops_; }

    const std::unordered_map<std::string, std::vector<ID>> &xLoops() const {
        return xLoops_;
    }

  private:
    template <class T> void recordAccess(const T &op) {
        if (inside_) {
            (isAfter_ ? part1Vars_ : part0Vars_).insert(op->var_);
        }
    }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class AddDimToVar : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    // VarDef ID -> for ID
    std::unordered_map<ID, std::vector<ID>> toAdd_;
    // for ID -> For
    std::unordered_map<ID, For> forMap_;

  public:
    AddDimToVar(const std::unordered_map<ID, std::vector<ID>> &toAdd)
        : toAdd_(toAdd) {}

  private:
    template <class T> T doAdd(T op) {
        if (toAdd_.count(def(op->var_)->id())) {
            for (auto &&loop : toAdd_.at(def(op->var_)->id())) {
                op->indices_.insert(
                    op->indices_.begin(),
                    makeFloorDiv(makeSub(makeVar(forMap_.at(loop)->iter_),
                                         forMap_.at(loop)->begin_),
                                 forMap_.at(loop)->step_));
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
    ID loop_;
    ID before_, after_;
    std::string suffix0_, suffix1_;
    std::unordered_map<ID, ID> ids0_, ids1_;
    std::unordered_set<std::string> varUses_;
    bool inside_ = false, isPart0_ = true, anyInside_ = false, isAfter_ = false;
    bool preserveFirst_, preserveSecond_;

  public:
    FissionFor(const ID &loop, const ID &before, const ID &after,
               bool preserveFirst, bool preserveSecond)
        : loop_(loop), before_(before), after_(after),
          preserveFirst_(preserveFirst), preserveSecond_(preserveSecond) {}

    const std::unordered_map<ID, ID> &ids0() const { return ids0_; }
    const std::unordered_map<ID, ID> &ids1() const { return ids1_; }

  private:
    void markNewId(const Stmt &op, bool isPart0);

    bool inPart() const {
        return inside_ && ((isPart0_ && !isAfter_) || (!isPart0_ && isAfter_));
    }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
};

std::pair<Stmt,
          std::pair<std::unordered_map<ID, ID>, std::unordered_map<ID, ID>>>
fission(const Stmt &ast, const ID &loop, FissionSide side, const ID &splitter,
        bool preserveFirstId, bool preserveSecondId);

} // namespace freetensor

#endif // FREE_TENSOR_FISSION_H
