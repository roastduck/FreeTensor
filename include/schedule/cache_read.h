#ifndef CACHE_READ_H
#define CACHE_READ_H

#include <mutator.h>

namespace ir {

class CacheRead : public Mutator {
    std::string stmt_, var_, fillStmt_, cacheVar_;
    Ref<Buffer> buffer_;
    std::vector<std::pair<uint64_t, Load>> loads_;
    bool inside_ = false;

  public:
    CacheRead(const std::string &stmt, const std::string &var)
        : stmt_(stmt), var_(var), fillStmt_(stmt_ + ".init"),
          cacheVar_(var + ".r") {}

    const std::string &fillStmt() const { return fillStmt_; }
    const std::string &cacheVar() const { return cacheVar_; }

  private:
    template <class T> Stmt doModify(const T &op) {
        if (op->id() == stmt_) {
            inside_ = true;
            auto ret = Mutator::visit(op);
            inside_ = false;

            // Make cache fill
            std::vector<Stmt> fill;
            for (auto &&item : loads_) {
                auto &&indices = item.second->indices_;
                fill.emplace_back(
                    makeStore("", cacheVar_, indices, makeLoad(var_, indices)));
            }

            fill.emplace_back(ret);
            ret = makeStmtSeq("", std::move(fill));
            ret =
                makeVarDef("", cacheVar_, std::move(*buffer_), std::move(ret));
            ret.template as<VarDefNode>()->buffer_->setAtype(AccessType::Cache);
            return ret;
        } else {
            return Mutator::visit(op);
        }
    }

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const StmtSeq &op) override { return doModify(op); }
    Stmt visit(const VarDef &op) override;
    Stmt visit(const Store &op) override { return doModify(op); }
    Stmt visit(const AddTo &op) override { return doModify(op); }
    Stmt visit(const For &op) override { return doModify(op); }
    Stmt visit(const If &op) override { return doModify(op); }
    Stmt visit(const Assert &op) override { return doModify(op); }
};

} // namespace ir

#endif // CACHE_READ_H
