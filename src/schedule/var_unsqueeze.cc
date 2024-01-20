#include <mutator.h>
#include <schedule.h>
#include <schedule/var_unsqueeze.h>

namespace freetensor {

namespace {

class VarUnsqueeze : public Mutator {
    ID defId_;
    int dim_;
    std::string var_;

  public:
    VarUnsqueeze(const ID &def, int dim) : defId_(def), dim_(dim) {}

  private:
    template <typename T> auto visitAcc(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (op->var_ == var_) {
            op->indices_.insert(op->indices_.begin() + dim_, makeIntConst(0));
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &_op) override {
        if (_op->id() == defId_) {
            if (dim_ < 0 ||
                dim_ > (int)_op->buffer_->tensor()->shape().size()) {
                throw InvalidSchedule("Invalid dimension " +
                                      std::to_string(dim_));
            }
            var_ = _op->name_;
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            var_.clear();
            op->buffer_->tensor()->shape().insert(
                op->buffer_->tensor()->shape().begin() + dim_, makeIntConst(1));
            return op;
        } else {
            return Mutator::visit(_op);
        }
    }

    Expr visit(const Load &op) override { return visitAcc(op); }
    Stmt visit(const Store &op) override { return visitAcc(op); }
    Stmt visit(const ReduceTo &op) override { return visitAcc(op); }
};

} // Anonymous namespace

Stmt varUnsqueeze(const Stmt &ast, const ID &def, int dim) {
    return VarUnsqueeze{def, dim}(ast);
}

void Schedule::varUnsqueeze(const ID &def, int dim) {
    beginTransaction();
    auto log = appendLog(
        MAKE_SCHEDULE_LOG(VarUnsqueeze, freetensor::varUnsqueeze, def, dim));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
