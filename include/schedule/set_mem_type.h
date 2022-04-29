#ifndef FREE_TENSOR_SET_MEM_TYPE_H
#define FREE_TENSOR_SET_MEM_TYPE_H

#include <unordered_map>

#include <mutator.h>

namespace freetensor {

class SetMemType : public Mutator {
    ID def_;
    MemType mtype_;
    std::unordered_map<ParallelScope, int> inScope_;
    bool found_ = false;

  public:
    SetMemType(const ID &def, MemType mtype) : def_(def), mtype_(mtype) {}

    bool found() const { return found_; }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt setMemType(const Stmt &ast, const ID &def, MemType mtype);

} // namespace freetensor

#endif // FREE_TENSOR_SET_MEM_TYPE_H
