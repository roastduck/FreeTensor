#ifndef CACHE_READ_H
#define CACHE_READ_H

#include <analyze/check_all_defined.h>
#include <except.h>
#include <mutator.h>

namespace ir {

class CacheRead : public Mutator {
    std::string stmt_, var_, fillStmt_, cacheVar_;
    Ref<Buffer> buffer_;
    std::vector<std::pair<uint64_t, Load>> loads_;
    std::unordered_set<std::string> defs_;
    bool inside_ = false;
    bool modified_ = false;

  public:
    CacheRead(const std::string &stmt, const std::string &var)
        : stmt_(stmt), var_(var), fillStmt_(stmt_ + ".init"),
          cacheVar_(var + ".r") {}

    const std::string &fillStmt() const { return fillStmt_; }
    const std::string &cacheVar() const { return cacheVar_; }
    bool modified() const { return modified_; }

  private:
    template <class T> Stmt recurseProxy(const T &op) {
        return Mutator::visit(op);
    }
    Stmt recurseProxy(const For &op) {
        defs_.insert(op->iter_);
        auto ret = Mutator::visit(op);
        defs_.erase(op->iter_);
        return ret;
    }
    Stmt recurseProxy(const VarDef &op) {
        if (op->name_ == var_) {
            buffer_ = op->buffer_;
        }
        defs_.insert(op->name_);
        auto ret = Mutator::visit(op);
        defs_.erase(op->name_);
        return ret;
    }

    template <class T> Stmt doModify(const T &op) {
        if (op->id() == stmt_) {
            inside_ = true;
            auto ret = recurseProxy(op);
            inside_ = false;

            if (loads_.empty()) {
                throw InvalidSchedule("No loads found from the specific "
                                      "variable in the given scope");
            }

            // Make cache fill
            std::vector<Stmt> fill;
            for (auto &&item : loads_) {
                if (!checkAllDefined(defs_, item.second)) {
                    throw InvalidSchedule(
                        "Using local variables defined in `stmt` to read `var` "
                        "is not supported");
                }
                auto &&indices = item.second->indices_;
                fill.emplace_back(
                    makeStore("", cacheVar_, indices, makeLoad(var_, indices)));
            }

            fill.emplace_back(ret);
            ret = makeStmtSeq("", std::move(fill));
            ret =
                makeVarDef("", cacheVar_, std::move(*buffer_), std::move(ret));
            ret.template as<VarDefNode>()->buffer_->setAtype(AccessType::Cache);
            modified_ = true;
            return ret;
        } else {
            return recurseProxy(op);
        }
    }

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const StmtSeq &op) override { return doModify(op); }
    Stmt visit(const VarDef &op) override { return doModify(op); }
    Stmt visit(const Store &op) override { return doModify(op); }
    Stmt visit(const AddTo &op) override { return doModify(op); }
    Stmt visit(const For &op) override { return doModify(op); }
    Stmt visit(const If &op) override { return doModify(op); }
    Stmt visit(const Assert &op) override { return doModify(op); }
};

} // namespace ir

#endif // CACHE_READ_H
