#ifndef FREE_TENSOR_MAKE_HEAP_ALLOC_H
#define FREE_TENSOR_MAKE_HEAP_ALLOC_H

#include <unordered_set>

#include <driver/target.h>
#include <func.h>
#include <mutator.h>

namespace freetensor {

class InsertAlloc : public Mutator {
    std::string var_;
    bool is_insert;

  public:
    InsertAlloc(const std::string &var) : var_(var), is_insert(true) {}

  protected:
    Stmt visit(const StmtSeq &op) override;
};

class InsertFree : public Mutator {
    std::string var_;
    bool is_insert;

  public:
    InsertFree(const std::string &var) : var_(var), is_insert(true) {}

  protected:
    Stmt visit(const StmtSeq &op) override;
};

class MakeHeapAlloc : public Mutator {
  private:
    bool inCublas_ = false;
    int for_depth = 0;
    bool inKernel() const;

  protected:
    Stmt visit(const VarDef &op) override;
    Stmt visit(const For &op) override;
    Stmt visit(const MatMul &op) override;
};

/**
 * For any variable with memory type of cpu/heap or gpu/global/heap, we allocate
 * memory for it as late as possible, and deallocate memory as early as
 * possible. Similar operations will not be performed on scalars cause of few
 * memory they used.
 */
Stmt makeHeapAlloc(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeHeapAlloc);

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_HEAP_ALLOC_H
