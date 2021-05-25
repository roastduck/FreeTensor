#include <algorithm>
#include <unordered_map>

#include <func.h>
#include <mutator.h>

namespace ir {

namespace {

class Func2Stmt : public Mutator {
    Func func_;
    std::string callSiteId_;
    const std::vector<std::string> &args_;

    std::unordered_map<std::string, std::string> replace_;

  public:
    Func2Stmt(const Func &func, const std::string callSiteId,
              const std::vector<std::string> &args)
        : func_(func), callSiteId_(callSiteId), args_(args) {}

  protected:
    Stmt
    visitStmt(const Stmt &op,
              const std::function<Stmt(const Stmt &)> &visitNode) override {
        auto ret = Mutator::visitStmt(op, visitNode);
        if (ret->id()[0] != '#') {
            ret->setId(callSiteId_ + "." + ret->id());
        }
        return ret;
    }

    Stmt visit(const VarDef &_op) override {
        if (_op->buffer_->atype() != AccessType::Cache) {
            auto it = std::find(func_->params_.begin(), func_->params_.end(),
                                _op->name_);
            if (it == func_->params_.end()) {
                throw InvalidProgram(
                    "I/O variable " + _op->name_ +
                    " should be in the signature of function " + func_->name_);
            }
            auto nth = it - func_->params_.begin();
            ASSERT(!replace_.count(_op->name_));
            replace_[_op->name_] = args_.at(nth);
            auto ret = (*this)(_op->body_);
            replace_.erase(_op->name_);
            return ret;
        } else {
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            op->name_ = callSiteId_ + "." + op->name_;
            return op;
        }
    }

    Stmt visit(const For &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::For);
        auto op = __op.as<ForNode>();
        op->iter_ = callSiteId_ + "." + op->iter_;
        return op;
    }

    Expr visit(const Var &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::Var);
        auto op = __op.as<VarNode>();
        op->name_ = callSiteId_ + "." + op->name_;
        return op;
    }

    Expr visit(const Load &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::Load);
        auto op = __op.as<LoadNode>();
        op->var_ = replace_.count(op->var_) ? replace_.at(op->var_)
                                            : callSiteId_ + "." + op->var_;
        return op;
    }

    Stmt visit(const Store &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::Store);
        auto op = __op.as<StoreNode>();
        op->var_ = replace_.count(op->var_) ? replace_.at(op->var_)
                                            : callSiteId_ + "." + op->var_;
        return op;
    }

    Stmt visit(const ReduceTo &_op) override {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
        auto op = __op.as<ReduceToNode>();
        op->var_ = replace_.count(op->var_) ? replace_.at(op->var_)
                                            : callSiteId_ + "." + op->var_;
        return op;
    }
};

} // Anonymous namespace

Stmt func2stmt(const Func &func, const std::vector<std::string> &args,
               const std::string &callSiteId) {
    return Func2Stmt(func, callSiteId, args)(func->body_);
}

} // namespace ir
