#ifndef FISSION_H
#define FISSION_H

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <mutator.h>

namespace ir {

class HoistVar : public Mutator {
    std::string loop_, after_, seqId_;
    std::unordered_set<std::string> part0Vars_, part1Vars_;
    std::vector<VarDef> defStack_;

    // var name -> loop id: which loops will a var cross during hoisting?
    std::unordered_map<std::string, std::vector<std::string>> xLoops_;

    bool inside_ = false, isAfter_ = false;

  public:
    HoistVar(const std::string &loop, const std::string &after)
        : loop_(loop), after_(after) {}

    const std::string &seqId() const { return seqId_; }

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
    virtual Stmt visit(const For &op) override;
    virtual Stmt visit(const StmtSeq &op) override;
    virtual Stmt visit(const VarDef &op) override;
    virtual Stmt visit(const Store &op) override;
    virtual Expr visit(const Load &op) override;
    virtual Stmt visit(const AddTo &op) override;
};

class AddDimToVar : public Mutator {
    // var name -> for ID
    std::unordered_map<std::string, std::vector<std::string>> toAdd_;
    // for ID -> For
    std::unordered_map<std::string, For> forMap_;

  public:
    AddDimToVar(
        const std::unordered_map<std::string, std::vector<std::string>> &toAdd)
        : toAdd_(toAdd) {}

  private:
    template <class T> T doAdd(T op) {
        if (toAdd_.count(op->var_)) {
            auto &&loops = toAdd_.at(op->var_);
            for (auto it = loops.rbegin(); it != loops.rend(); it++) {
                op->indices_.emplace_back(makeVar(forMap_.at(*it)->iter_));
            }
        }
        return op;
    }

  protected:
    virtual Stmt visit(const For &op) override;
    virtual Stmt visit(const VarDef &op) override;
    virtual Stmt visit(const Store &op) override;
    virtual Stmt visit(const AddTo &op) override;
    virtual Expr visit(const Load &op) override;
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
