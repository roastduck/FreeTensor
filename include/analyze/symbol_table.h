#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include <type_traits>
#include <unordered_map>

#include <stmt.h>

namespace ir {

class SymbolTableInterface {
  public:
    virtual const bool hasDef(const std::string &name) const = 0;
    virtual const VarDef &def(const std::string &name) const = 0;
    virtual const Ref<Buffer> &buffer(const std::string &name) const = 0;
    virtual void pushDef(const VarDef &op) = 0;
    virtual void popDef(const VarDef &op) = 0;
};

/**
 * A symbol table context for Visitor or Mutator
 *
 * Inherit this class to use. E.g., inherit SymbolTable<Visitor> or
 * SymbolTable<Mutator>
 *
 * This class will automatically maintains the symbol table if the sub-class
 * calls visit(VarDef), which is the suggested usage
 *
 * However, in some cases, this is impossible, e.g., when the sub-class needs to
 * recurse into different sub-trees manually. In these cases, the sub-class
 * should explicitly call the pushDef and popDef method
 */
template <class BaseClass>
class SymbolTable : public BaseClass, public SymbolTableInterface {
    std::unordered_map<std::string, VarDef> defs_;

  public:
    const bool hasDef(const std::string &name) const override {
        return defs_.count(name);
    }

    const VarDef &def(const std::string &name) const override {
        return defs_.at(name);
    }

    const Ref<Buffer> &buffer(const std::string &name) const override {
        return def(name)->buffer_;
    }

    void pushDef(const VarDef &op) override {
        if (defs_.count(op->name_)) {
            throw InvalidProgram("Nested VarDef with the same name \"" +
                                 op->name_ + "\"is not allowed");
        }
        defs_[op->name_] = op;
    }

    void popDef(const VarDef &op) override { defs_.erase(op->name_); }

  protected:
    using BaseClass::visit;

    typename BaseClass::StmtRetType visit(const VarDef &op) {

        if constexpr (std::is_same_v<typename BaseClass::StmtRetType, void>) {
            for (auto &&dim : op->buffer_->tensor().shape()) {
                (*this)(dim);
            }
            if (op->sizeLim_.isValid()) {
                (*this)(op->sizeLim_);
            }

            pushDef(op);
            (*this)(op->body_);
            popDef(op);
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

            pushDef(op);
            auto body = (*this)(op->body_);
            popDef(op);

            return COPY_DEBUG_INFO(makeVarDef(op->id(), op->name_, std::move(b),
                                              std::move(sizeLim),
                                              std::move(body), op->pinned_),
                                   op);
        }
    }
};

} // namespace ir

#endif // SYMBOL_TABLE_H
