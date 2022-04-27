#ifndef FREE_TENSOR_GPU_MAKE_SYNC_H
#define FREE_TENSOR_GPU_MAKE_SYNC_H

#include <unordered_set>

#include <func.h>
#include <math/bounds.h>
#include <mutator.h>
#include <visitor.h>

namespace freetensor {

namespace gpu {

struct ThreadInfo {
    For loop_;
    bool inWarp_;
};

class FindAllThreads : public Visitor {
    int warpSize_ = 32; // TODO: Adjust to different arch
    int thx_ = 1, thy_ = 1, thz_ = 1;
    std::vector<ThreadInfo> results_;

  public:
    const std::vector<ThreadInfo> &results() const { return results_; }

  protected:
    void visit(const For &op) override;
};

class CopyParts : public Mutator {
    Expr cond_;
    const std::vector<Stmt> &splitters_;
    std::unordered_set<Stmt> fullParts_;

  public:
    CopyParts(const Expr &cond, const std::vector<Stmt> &splitters)
        : cond_(cond), splitters_(splitters) {}

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
    Stmt visit(const Assert &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const StmtSeq &op) override;
};

struct CrossThreadDep {
    Stmt later_, earlier_, lcaStmt_, lcaLoop_;
    bool inWarp_;
    bool visiting_, synced_;
};

class MakeSync : public Mutator {
    typedef Mutator BaseClass;

    Stmt root_;
    std::vector<CrossThreadDep> deps_;
    std::unordered_map<ID, Stmt> syncBeforeFor_;
    std::unordered_map<ID, std::vector<Stmt>> branchSplittersThen_,
        branchSplittersElse_;

  public:
    MakeSync(const Stmt root, std::vector<CrossThreadDep> &&deps)
        : root_(root), deps_(std::move(deps)) {}

  private:
    void markSyncForSplitting(const Stmt &stmtInTree, const Stmt &sync);

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
};

Stmt makeSync(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeSync)

} // namespace gpu

} // namespace freetensor

#endif // FREE_TENSOR_GPU_MAKE_SYNC_H
