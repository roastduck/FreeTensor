#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include <type_traits>
#include <unordered_map>

#include <stmt.h>

namespace ir {

template <class BaseClass> class SymbolTable : public BaseClass {
    std::unordered_map<std::string, VarDef> defs_;

  public:
    const bool hasDef(const std::string &name) const {
        return defs_.count(name);
    }

    const VarDef &def(const std::string &name) const { return defs_.at(name); }

    const Ref<Buffer> &buffer(const std::string &name) const {
        return def(name)->buffer_;
    }

  protected:
    using BaseClass::visit;

    typename BaseClass::StmtRetType visit(const VarDef &op) {
        if (defs_.count(op->name_)) {
            throw InvalidProgram("Nested VarDef with the same name \"" +
                                 op->name_ + "\"is not allowed");
        }

        if constexpr (std::is_same_v<typename BaseClass::StmtRetType, void>) {
            for (auto &&dim : op->buffer_->tensor().shape()) {
                (*this)(dim);
            }
            if (op->sizeLim_.isValid()) {
                (*this)(op->sizeLim_);
            }

            defs_[op->name_] = op;
            (*this)(op->body_);
            defs_.erase(op->name_);
        } else {
            std::vector<SubTree<ExprNode>> shape;
            shape.reserve(op->buffer_->tensor().shape().size());
            for (auto &&dim : op->buffer_->tensor().shape()) {
                shape.emplace_back((*this)(dim));
            }
            Tensor t(std::move(shape), op->buffer_->tensor().dtype());
            Buffer b(std::move(t), op->buffer_->atype(), op->buffer_->mtype());
            Expr sizeLim =
                op->sizeLim_.isValid() ? (*this)(op->sizeLim_) : nullptr;

            defs_[op->name_] = op;
            auto body = (*this)(op->body_);
            defs_.erase(op->name_);

            return COPY_DEBUG_INFO(makeVarDef(op->id(), op->name_, std::move(b),
                                              std::move(sizeLim),
                                              std::move(body), op->pinned_),
                                   op);
        }
    }
};

} // namespace ir

#endif // SYMBOL_TABLE_H
