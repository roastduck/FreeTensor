#ifndef FREE_TENSOR_SET_MEM_TYPE_H
#define FREE_TENSOR_SET_MEM_TYPE_H

#include <unordered_map>

#include <analyze/all_uses.h>
#include <analyze/symbol_table.h>
#include <mutator.h>
#include <pass/const_fold.h>
#include <visitor.h>

namespace freetensor {

class ThrowIndirectAccess : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    ID def_;

  public:
    ThrowIndirectAccess(const ID &def) : def_(def) {}

  private:
    template <typename T> void visitAcc(const T &op) {
        BaseClass::visit(op);
        if (def(op->var_)->id() == def_) {
            for (auto &&dim : op->indices_) {
                if (!allReads(dim, true).empty()) {
                    goto err;
                }
                for (auto &&iter : allIters(dim)) {
                    if (!constFold(loop(iter)->len_)->isConst()) {
                        goto err;
                    }
                }
                continue;
            err:
                throw InvalidSchedule("Encountering indirect access in " +
                                      toString(op) +
                                      " (treated as exception because "
                                      "`reject_indirect_access` is set)");
            }
        }
    }

  public:
    using BaseClass::visit;
    void visit(const Load &op) override { visitAcc(op); }
    void visit(const Store &op) override { visitAcc(op); }
    void visit(const ReduceTo &op) override { visitAcc(op); }
};

class SetMemType : public Mutator {
    ID def_;
    MemType mtype_;

  public:
    SetMemType(const ID &def, MemType mtype) : def_(def), mtype_(mtype) {}

  protected:
    Stmt visit(const VarDef &op) override;
};

Stmt setMemType(const Stmt &ast, const ID &def, MemType mtype,
                bool rejectIndirectAccess);

} // namespace freetensor

#endif // FREE_TENSOR_SET_MEM_TYPE_H
