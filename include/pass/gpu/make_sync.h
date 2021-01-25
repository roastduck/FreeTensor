#ifndef GPU_MAKE_SYNC_H
#define GPU_MAKE_SYNC_H

#include <mutator.h>

namespace ir {

namespace gpu {

class MakeSync : public Mutator {
    int warpSize = 32; // TODO: Adjust to different arch

    bool warpSynced = true, threadsSynced = true;
    int thx = 1, thy = 1, thz = 1;

  protected:
    Stmt visit(const For &op) override;

    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

Stmt makeSync(const Stmt &op);

} // namespace gpu

} // namespace ir

#endif // GPU_MAKE_SYNC_H
