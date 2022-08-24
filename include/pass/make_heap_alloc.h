#ifndef FREE_TENSOR_MAKE_HEAP_ALLOC_H
#define FREE_TENSOR_MAKE_HEAP_ALLOC_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>
#include <pass/const_fold.h>

namespace freetensor {

class InsertAlloc : public Mutator {
    std::string var_;
    bool isOuterMost_ = true, delayed_ = false;

  public:
    InsertAlloc(const std::string &var) : var_(var) {}

    bool delayed() const { return delayed_; }

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override { return op; }
    Stmt visit(const If &op) override { return op; }
};

class InsertFree : public Mutator {
    std::string var_;
    bool isOuterMost_ = true, madeEarly_ = false;

  public:
    InsertFree(const std::string &var) : var_(var) {}

    bool madeEarly() const { return madeEarly_; }

  protected:
    Stmt visit(const StmtSeq &op) override;
    Stmt visit(const For &op) override { return op; }
    Stmt visit(const If &op) override { return op; }
};

class MakeHeapAlloc : public ConstFold {
    // Inherit ConstFold for determine dynamic sizes
    typedef ConstFold BaseClass;

    bool inKernel_ = false;

  private:
    bool inKernel() const;
    bool isDynamicSized(const VarDef &op) const;

  protected:
    using BaseClass::visit;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const BSPScope &op) override;
    Stmt visit(const MatMul &op) override { return op; }
};

/**
 * Insert Alloc and Free node for heap-allocated varialbes, and turn
 * stack-allocated variables to heap-allocated is beneficial
 *
 * Varaibles with `MemType::CPUHeap` or `MemType::GPUGlobalHeap` are allocated
 * on the heap by some allocators. Other variables are allocated in a stack
 * manner, not allocated by an allocator, but they may also be on the heap
 * physically
 *
 * Dynamic-sized variables cannot be allocated on the stack, so turn them to be
 * allocated on the heap
 *
 * For other variables, if we can delay its allocation or free them earlier, we
 * allocate them on the heap. The delation or making early will not cross any
 * control flow
 *
 * This transformation is not applied to scalars
 */
Stmt makeHeapAlloc(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeHeapAlloc);

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_HEAP_ALLOC_H
