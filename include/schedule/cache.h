#ifndef CACHE_H
#define CACHE_H

#include <analyze/comp_access_bound.h>
#include <mutator.h>

namespace ir {

class MakeCacheVar : public Mutator {
    std::string stmt_, oldVar_, newVar_, oldDef_, newDef_;
    MemType mtype_;
    VarDef def_;
    bool inStmt_ = false;

  public:
    MakeCacheVar(const std::string &stmt, const std::string &oldVar,
                 MemType mtype)
        : stmt_(stmt), oldVar_(oldVar), newVar_(oldVar + ".c"), mtype_(mtype) {}

    const std::string &newVar() const { return newVar_; }
    const std::string &oldDef() const { return oldDef_; }
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
    std::string stmt_, oldVar_, newVar_, oldDef_;
    std::string fillStmt_, flushStmt_;
    const AccessBound &rRange_, &wRange_;
    VarDef def_;

  public:
    MakeFillAndFlush(const std::string &stmt, const std::string &oldVar,
                     const std::string &newVar, const std::string &oldDef,
                     const AccessBound &rRange, const AccessBound &wRange)
        : stmt_(stmt), oldVar_(oldVar), newVar_(newVar), oldDef_(oldDef),
          rRange_(rRange), wRange_(wRange) {}

    const std::string &fillStmt() const { return fillStmt_; }
    const std::string &flushStmt() const { return flushStmt_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const VarDef &op) override;
};

class MakeInitAndReduce : public Mutator {
    std::string stmt_, oldVar_, newVar_, oldDef_, newDef_;
    std::string initStmt_, reduceStmt_;
    const AccessBound &range_;
    VarDef def_;
    ReduceTo reduce_;
    bool inNewVar_ = false;

  public:
    MakeInitAndReduce(const std::string &stmt, const std::string &oldVar,
                      const std::string &newVar, const std::string &oldDef,
                      const std::string &newDef, const AccessBound &range)
        : stmt_(stmt), oldVar_(oldVar), newVar_(newVar), oldDef_(oldDef),
          newDef_(newDef), range_(range) {}

    const std::string &initStmt() const { return initStmt_; }
    const std::string &reduceStmt() const { return reduceStmt_; }

  protected:
    Stmt visitStmt(const Stmt &op,
                   const std::function<Stmt(const Stmt &)> &visitNode) override;
    Stmt visit(const VarDef &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const Store &op) override;
    Expr visit(const Load &op) override;
};

} // namespace ir

#endif // CACHE_H
