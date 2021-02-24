#ifndef CACHE_H
#define CACHE_H

#include <analyze/comp_access_bound.h>
#include <mutator.h>

namespace ir {

class MakeCacheVar : public Mutator {
    std::string stmt_, oldVar_, newVar_, newDef_;
    MemType mtype_;
    Ref<Buffer> oldBuffer_;
    bool inStmt_ = false;

  public:
    MakeCacheVar(const std::string &stmt, const std::string &oldVar,
                 MemType mtype)
        : stmt_(stmt), oldVar_(oldVar), newVar_(oldVar + ".c"), mtype_(mtype) {}

    const std::string &newVar() const { return newVar_; }
    const std::string &newDef() const { return newDef_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const VarDef &op) override;
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
};

class MakeFillAndFlush : public Mutator {
    std::string stmt_, oldVar_, newVar_, newDef_, fillStmt_, flushStmt_;
    const std::unordered_map<std::string, AccessBound> &rRange_, &wRange_;
    int nDim_ = -1;

  public:
    MakeFillAndFlush(const std::string &stmt, const std::string &oldVar,
                     const std::string &newVar, const std::string &newDef,
                     const std::unordered_map<std::string, AccessBound> &rRange,
                     const std::unordered_map<std::string, AccessBound> &wRange)
        : stmt_(stmt), oldVar_(oldVar), newVar_(newVar), newDef_(newDef),
          rRange_(rRange), wRange_(wRange) {}

    const std::string &fillStmt() const { return fillStmt_; }
    const std::string &flushStmt() const { return flushStmt_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const VarDef &op) override;
};

} // namespace ir

#endif // CACHE_H
