#ifndef CACHE_WRITE_H
#define CACHE_WRITE_H

#include <analyze/check_all_defined.h>
#include <analyze/hash.h>
#include <except.h>
#include <mutator.h>

namespace ir {

class CacheWrite : public Mutator {
    std::string stmt_, var_, initStmt_, flushStmt_, cacheVar_;
    MemType mtype_;
    Ref<Buffer> buffer_;
    std::vector<Stmt> stores_;
    std::unordered_set<std::string> defs_;
    bool inside_ = false;
    bool modified_ = false;

  public:
    CacheWrite(const std::string &stmt, const std::string &var, MemType mtype)
        : stmt_(stmt), var_(var), initStmt_(stmt_ + ".init"),
          flushStmt_(stmt_ + ".final"), cacheVar_(var + ".w"), mtype_(mtype) {}

    const std::string &initStmt() const { return initStmt_; }
    const std::string &flushStmt() const { return flushStmt_; }
    const std::string &cacheVar() const { return cacheVar_; }
    bool modified() const { return modified_; }

  private:
    Expr makeNeutralVal(DataType dtype, ReduceOp op);

    template <class T, class U> bool sameIndices(const T &lhs, const U &rhs) {
        ASSERT(lhs->indices_.size() == rhs->indices_.size());
        for (size_t i = 0, iEnd = lhs->indices_.size(); i < iEnd; i++) {
            if (getHash(lhs->indices_[i]) != getHash(rhs->indices_[i])) {
                return false;
            }
        }
        return true;
    }

    template <class T> bool sameIndicesProxy(const T &lhs, const Stmt &rhs) {
        if (rhs->nodeType() == ASTNodeType::Store) {
            return sameIndices(lhs, rhs.as<StoreNode>());
        } else if (rhs->nodeType() == ASTNodeType::ReduceTo) {
            return sameIndices(lhs, rhs.as<ReduceToNode>());
        } else {
            ASSERT(false);
        }
    }

    template <class T> Stmt recurseProxy(const T &op) {
        return Mutator::visit(op);
    }
    Stmt recurseProxy(const Store &op) { return visitStoreLike(op); }
    Stmt recurseProxy(const ReduceTo &op) { return visitStoreLike(op); }
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

    template <class T> Stmt visitStoreLike(const Ref<T> &_op) {
        auto op = Mutator::visit(_op).template as<T>();
        if (inside_ && op->var_ == var_) {
            for (auto &&item : stores_) {
                if (sameIndicesProxy(op, item)) {
                    goto done;
                }
            }
            stores_.emplace_back(_op);
        done:
            op->var_ = cacheVar_;
        }
        return op;
    }

    template <class T> Stmt doModify(const T &op) {
        if (op->id() == stmt_) {
            inside_ = true;
            auto ret = recurseProxy(op);
            inside_ = false;

            if (stores_.empty()) {
                throw InvalidSchedule(
                    "no stores to the specified variable in the given scope");
            }

            // Make cache init and flush
            Stmt init;
            std::vector<Stmt> flush;
            for (auto &&item : stores_) {
                if (!checkAllDefined(defs_, item)) {
                    throw InvalidSchedule("Using local variables defined in "
                                          "`stmt` to write `var` "
                                          "is not supported");
                }
                switch (item->nodeType()) {
                case ASTNodeType::Store: {
                    auto &&indices = item.as<StoreNode>()->indices_;
                    flush.emplace_back(makeStore("", var_, indices,
                                                 makeLoad(cacheVar_, indices)));
                    break;
                }
                case ASTNodeType::ReduceTo: {
                    auto &&indices = item.as<ReduceToNode>()->indices_;
                    auto reduceOp = item.as<ReduceToNode>()->op_;
                    if (init.isValid()) {
                        throw InvalidSchedule(
                            "Only one reduction statement (ReduceTo node) is "
                            "allowed in a cached region. This is to ensure the "
                            "commutative law of the reduciton");
                    }
                    init = makeStore(
                        "", cacheVar_, indices,
                        makeNeutralVal(buffer_->tensor().dtype(), reduceOp));
                    flush.emplace_back(
                        makeReduceTo("", var_, indices, reduceOp,
                                     makeLoad(cacheVar_, indices),
                                     item.as<ReduceToNode>()->atomic_));
                    break;
                }
                default:
                    ASSERT(false);
                }
            }

            init = init.isValid() ? init : makeStmtSeq("", {});
            auto f = flush.size() == 1 ? flush[0]
                                       : makeStmtSeq("", std::move(flush));
            init->setId(initStmt_);
            f->setId(flushStmt_);
            ret = makeStmtSeq("", {init, ret, f});
            ret = makeVarDef("", cacheVar_, *buffer_, std::move(ret));
            ret.template as<VarDefNode>()->buffer_->setAtype(AccessType::Cache);
            ret.template as<VarDefNode>()->buffer_->setMtype(mtype_);
            modified_ = true;
            return ret;
        } else {
            return recurseProxy(op);
        }
    }

  protected:
    Stmt visit(const Store &op) override { return doModify(op); }
    Stmt visit(const ReduceTo &op) override { return doModify(op); }
    Stmt visit(const StmtSeq &op) override { return doModify(op); }
    Stmt visit(const VarDef &op) override { return doModify(op); }
    Stmt visit(const For &op) override { return doModify(op); }
    Stmt visit(const If &op) override { return doModify(op); }
    Stmt visit(const Assert &op) override { return doModify(op); }
};

} // namespace ir

#endif // CACHE_WRITE_H
