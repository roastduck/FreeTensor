#ifndef GPU_MAKE_SYNC_H
#define GPU_MAKE_SYNC_H

#include <cursor.h>
#include <func.h>
#include <math/bounds.h>

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

class CopyPart : public Mutator {
    Stmt begin_, end_;
    bool begun_, ended_;
    std::vector<VarDef> splittedDefs_; // From inner to outer

  public:
    CopyPart(const Stmt &begin, const Stmt &end)
        : begin_(begin), end_(end), begun_(!begin.isValid()), ended_(false) {}

    const std::vector<VarDef> &splittedDefs() const { return splittedDefs_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

struct CrossThreadDep {
    Cursor later_, earlier_, lcaStmt_, lcaLoop_;
    bool inWarp_;
    bool visiting_, synced_;
};

class MakeSync : public MutatorWithCursor {
    Stmt root_;
    std::vector<CrossThreadDep> deps_;
    std::unordered_map<std::string, Stmt> syncBeforeFor_;
    std::unordered_map<std::string, std::vector<Stmt>> branchSplittersThen_,
        branchSplittersElse_;

  public:
    MakeSync(const Stmt root, std::vector<CrossThreadDep> &&deps)
        : root_(root), deps_(std::move(deps)) {}

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
};

Stmt makeSync(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeSync)

} // namespace gpu

} // namespace ir

#endif // GPU_MAKE_SYNC_H
