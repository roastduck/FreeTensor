#include <hash.h>
#include <mutator.h>
#include <schedule.h>
#include <schedule/var_squeeze.h>

namespace freetensor {

namespace {

class VarSqueeze : public Mutator {
    ID defId_;
    int dim_;
    std::string var_;

  public:
    VarSqueeze(const ID &def, int dim) : defId_(def), dim_(dim) {}

  private:
    template <typename T> auto visitAcc(const T &_op) {
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == _op->nodeType());
        auto op = __op.template as<typename T::Object>();
        if (op->var_ == var_) {
            op->indices_.erase(op->indices_.begin() + dim_);
        }
        return op;
    }

  protected:
    Stmt visit(const VarDef &_op) override {
        if (_op->id() == defId_) {
            if (dim_ < 0 ||
                dim_ >= (int)_op->buffer_->tensor()->shape().size()) {
                throw InvalidSchedule("Invalid dimension " +
                                      std::to_string(dim_));
            }
            var_ = _op->name_;
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            var_.clear();
            if (!HashComparator{}(op->buffer_->tensor()->shape()[dim_],
                                  makeIntConst(1))) {
                throw InvalidSchedule("Dimension " + std::to_string(dim_) +
                                      " is not 1-lengthed");
            }
            op->buffer_->tensor()->shape().erase(
                op->buffer_->tensor()->shape().begin() + dim_);
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

Stmt varSqueeze(const Stmt &ast, const ID &def, int dim) {
    return VarSqueeze{def, dim}(ast);
}

void Schedule::varSqueeze(const ID &def, int dim) {
    beginTransaction();
    auto log = appendLog(
        MAKE_SCHEDULE_LOG(VarSqueeze, freetensor::varSqueeze, def, dim));
    try {
        applyLog(log);
        commitTransaction();
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
