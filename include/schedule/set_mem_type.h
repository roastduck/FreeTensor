#ifndef SET_MEM_TYPE_H
#define SET_MEM_TYPE_H

#include <unordered_map>

#include <mutator.h>

namespace ir {

class SetMemType : public Mutator {
    std::string def_;
    MemType mtype_;
    std::unordered_map<std::string, int> inScope_;
    bool found_ = false;

  public:
    SetMemType(const std::string &def, MemType mtype)
        : def_(def), mtype_(mtype) {}

    bool found() const { return found_; }

  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt setMemType(const Stmt &ast, const std::string &def, MemType mtype);

} // namespace ir

#endif // SET_MEM_TYPE_H
