#ifndef FREE_TENSOR_MAKE_HEAP_ALLOC_H
#define FREE_TENSOR_MAKE_HEAP_ALLOC_H

#include <unordered_set>

#include <func.h>
#include <mutator.h>

namespace freetensor {

class InsertAlloc: public Mutator {
    std::string var_;

  public:
    InsertAlloc(const std::string &var) : var_(var) {}
  
  protected:
    Stmt visit(const StmtSeq &op) override;
};

class InsertFree: public Mutator {
    std::string var_;
  
  public:
    InsertFree(const std::string &var) : var_(var) {}
  
  protected:
    Stmt visit(const StmtSeq &op) override;
};

class InsertAllocFree: public Mutator {
    std::string var_;
  
  public:
    InsertAllocFree(const std::string &var) : var_(var) {}

  protected:
    Stmt visit(const StmtSeq &op) override;
};

class MakeHeapAlloc : public Mutator {
  protected:
    Stmt visit(const VarDef &op) override;
};

Stmt makeHeapAlloc(const Stmt &op);
	
DEFINE_PASS_FOR_FUNC(makeHeapAlloc);

} // namespace freetensor

#endif // FREE_TENSOR_MAKE_HEAP_ALLOC_H
