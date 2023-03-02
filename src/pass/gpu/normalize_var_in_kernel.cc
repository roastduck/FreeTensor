#include <container_utils.h>
#include <pass/gpu/normalize_var_in_kernel.h>
#include <pass/rename_var.h>
#include <pass/simplify.h>
#include <pass/z3_simplify.h>

namespace freetensor {

namespace gpu {

namespace {

class CountNames : public Visitor {
    std::unordered_map<std::string, int> nameCnt_;

  public:
    const auto &nameCnt() const { return nameCnt_; }

  protected:
    void visit(const VarDef &op) override {
        Visitor::visit(op);
        nameCnt_[op->name_]++;
    }

    void visit(const For &op) override {
        Visitor::visit(op);
        nameCnt_[op->iter_]++;
    }
};

std::unordered_map<std::string, int> countNames(const Stmt &s) {
    CountNames visitor;
    visitor(s);
    return visitor.nameCnt();
}

std::string getNewName(const std::string &oldName,
                       const std::unordered_set<std::string> &used) {
    for (int i = 1;; i++) {
        if (auto name = oldName + "." + std::to_string(i); !used.count(name)) {
            return name;
        }
    }
}

} // Anonymous namespace

Stmt NormalizeVarInKernel::visit(const VarDef &_op) {
    if (inKernel_) {
        auto __op = BaseClass::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();

        for (auto &dim : op->buffer_->tensor()->shape()) {
            Expr newDim;
            for (auto &&b : unique_.getDefinedUpper(
                     dim, ranges::to<std::unordered_set>(legalNames_))) {
                newDim = newDim.isValid() ? makeMin(std::move(newDim), b.expr())
                                          : b.expr();
            }
            if (!newDim.isValid()) {
                throw InvalidProgram(
                    "The shape of " + toString(op->id()) + " " + op->name_ +
                    " should be able to be determined outside a CUDA kernel");
            }
            dim = std::move(newDim);
        }

        if (op->buffer_->mtype() == MemType::GPUGlobalHeap ||
            op->buffer_->mtype() == MemType::GPUGlobal) {
            // Hoist so we are able to turn it into GPUGlobalHeap and insert
            // Alloc and Free. We don't use `hoist_selected_var` here because of
            // (compile-time) performance issue: A kernel is often quite large.
            // It is too slow to hoist these `VarDef`s all the way to outside a
            // kernel
            VarDef renamed = op;
            if (nameCntInKernel_.at(op->name_) > 1) {
                auto _renamed = renameVar(
                    op, op->name_, getNewName(op->name_, usedNamesInKernel_));
                ASSERT(_renamed->nodeType() == ASTNodeType::VarDef);
                renamed = _renamed.as<VarDefNode>();
                usedNamesInKernel_.insert(renamed->name_);
            }
            varsToHoist_.emplace_back(renamed);
            return renamed->body_;
        } else {
            return op;
        }
    } else {
        legalNames_.emplace_back(_op->name_);
        auto ret = BaseClass::visit(_op);
        legalNames_.pop_back();
        return ret;
    }
}

Stmt NormalizeVarInKernel::visit(const For &op) {
    if (!inKernel_ &&
        std::holds_alternative<CUDAScope>(op->property_->parallel_)) {
        nameCntInKernel_ = countNames(op);
        usedNamesInKernel_ =
            uni(this->names(),
                ranges::to<std::unordered_set>(nameCntInKernel_ | views::keys));

        inKernel_ = true;
        auto ret = BaseClass::visit(op);
        inKernel_ = false;

        for (auto &&def : varsToHoist_) {
            auto newRet = def;
            newRet->body_ = ret;
            ret = std::move(newRet);
        }
        varsToHoist_.clear();
        usedNamesInKernel_.clear();
        nameCntInKernel_.clear();
        return ret;
    } else {
        legalNames_.emplace_back(op->iter_);
        auto ret = BaseClass::visit(op);
        legalNames_.pop_back();
        return ret;
    }
}

Stmt normalizeVarInKernel(const Stmt &_op) {
    auto op = NormalizeVarInKernel{}(_op);
    return simplify(z3Simplify(op));
}

} // namespace gpu

} // namespace freetensor
