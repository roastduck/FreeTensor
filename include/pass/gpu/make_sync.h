#ifndef GPU_MAKE_SYNC_H
#define GPU_MAKE_SYNC_H

#include <unordered_set>

#include <cursor.h>
#include <func.h>
#include <math/bounds.h>
#include <mutator.h>

namespace ir {

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

struct CrossThreadDep {
    Cursor later_, earlier_, lcaLoop_;
    bool inWarp_;
    bool visiting_, synced_;
};

class MakeSync : public Mutator {
    std::vector<CrossThreadDep> deps_;
    StmtSeq whereToInsert_;
    std::unordered_set<StmtSeq> needSyncThreads_, needSyncWarp_;

  public:
    MakeSync(std::vector<CrossThreadDep> &&deps) : deps_(std::move(deps)) {}

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override;
};

Stmt makeSync(const Stmt &op);

inline Func makeSync(const Func &func) {
    return makeFunc(func->name_, func->params_, makeSync(func->body_));
}

} // namespace gpu

} // namespace ir

#endif // GPU_MAKE_SYNC_H
