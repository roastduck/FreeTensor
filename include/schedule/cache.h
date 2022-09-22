#ifndef FREE_TENSOR_CACHE_H
#define FREE_TENSOR_CACHE_H

#include <analyze/comp_access_bound.h>
#include <analyze/symbol_table.h>
#include <mutator.h>

namespace freetensor {

class MakeCacheVar : public Mutator {
    ID stmt_;
    std::string oldVar_, newVar_;
    ID oldDef_, newDef_;
    MemType mtype_;
    VarDef def_;
    bool inStmt_ = false;

  public:
    MakeCacheVar(const ID &stmt, const std::string &oldVar, MemType mtype,
                 bool isReduction)
        : stmt_(stmt), oldVar_(oldVar), mtype_(mtype) {
        newVar_ = oldVar_ + (isReduction ? ".r" : ".c");
        switch (mtype) {
        case MemType::GPULocal:
            newVar_ += ".local";
            break;
        case MemType::GPUShared:
            newVar_ += ".shared";
            break;
        case MemType::GPUGlobal:
            newVar_ += ".global";
            break;
        case MemType::GPUWarp:
            newVar_ += ".warp";
        default:;
        }
    }

    const std::string &newVar() const { return newVar_; }
    const ID &oldDef() const { return oldDef_; }
    const ID &newDef() const { return newDef_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class MakeFillAndFlush : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    ID stmt_;
    std::string oldVar_, newVar_;
    ID oldDef_, fillStmt_, flushStmt_;
    const AccessBound &rwRange_, &wRange_;
    VarDef def_;

  public:
    MakeFillAndFlush(const ID &stmt, const std::string &oldVar,
                     const std::string &newVar, const ID &oldDef,
                     const AccessBound &rwRange, const AccessBound &wRange)
        : stmt_(stmt), oldVar_(oldVar), newVar_(newVar), oldDef_(oldDef),
          rwRange_(rwRange), wRange_(wRange) {}

    const ID &fillStmt() const { return fillStmt_; }
    const ID &flushStmt() const { return flushStmt_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const VarDef &op) override;
};

class MakeInitAndReduce : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

    ID stmt_;
    std::string oldVar_, newVar_;
    ID oldDef_, newDef_, initStmt_, reduceStmt_;
    const AccessBound &range_;
    VarDef def_;
    ReduceTo reduce_;
    bool inNewVar_ = false;

  public:
    MakeInitAndReduce(const ID &stmt, const std::string &oldVar,
                      const std::string &newVar, const ID &oldDef,
                      const ID &newDef, const AccessBound &range)
        : stmt_(stmt), oldVar_(oldVar), newVar_(newVar), oldDef_(oldDef),
          newDef_(newDef), range_(range) {}

    const ID &initStmt() const { return initStmt_; }
    const ID &reduceStmt() const { return reduceStmt_; }

  protected:
    Stmt visitStmt(const Stmt &op) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
};

std::pair<Stmt, std::tuple<ID, ID, std::string, ID>>
cache(const Stmt &ast, const ID &stmt, const std::string &var, MemType mtype);

std::pair<Stmt, std::tuple<ID, ID, std::string, ID>>
cacheReduction(const Stmt &ast, const ID &stmt, const std::string &var,
               MemType mtype);

} // namespace freetensor

#endif // FREE_TENSOR_CACHE_H
