#include <codegen/native_code.h>
#include <visitor.h>

namespace freetensor {

namespace {

class FindSignatureTypes : public Visitor {
    std::vector<NativeCodeParam> params_;
    std::vector<NativeCodeRet> returns_;

  public:
    FindSignatureTypes(const std::vector<FuncParam> &funcParams,
                       const std::vector<FuncRet> &funcReturns) {
        params_.reserve(funcParams.size());
        for (auto &&funcParam : funcParams) {
            params_.emplace_back(funcParam.name_, std::nullopt,
                                 AccessType::Bypass, std::nullopt,
                                 funcParam.closure_, funcParam.updateClosure_);
        }

        returns_.reserve(funcReturns.size());
        for (auto &&funcReturn : funcReturns) {
            returns_.emplace_back(funcReturn.name_, funcReturn.dtype_,
                                  funcReturn.closure_,
                                  funcReturn.returnClosure_);
        }
    }

    const auto &params() const { return params_; }
    const auto &returns() const { return returns_; }

    void visit(const VarDef &op) override {
        Visitor::visit(op);
        if (op->buffer_->atype() != AccessType::Bypass) {
            for (auto &&param : params_) {
                if (param.name_ == op->name_) {
                    if (param.atype_ != AccessType::Bypass) {
                        throw InvalidProgram(
                            "Name " + op->name_ +
                            " should be unique in the AST as a paramerter");
                    }
                    param.dtype_ = op->buffer_->tensor()->dtype();
                    param.atype_ = op->buffer_->atype();
                    param.mtype_ = op->buffer_->mtype();
                }
            }
        }
    }
};

} // Anonymous namespace

std::ostream &operator<<(std::ostream &os, const NativeCodeParam &p) {
    os << p.name_ << ":";
    if (p.dtype_.has_value()) {
        os << " " << *p.dtype_;
    }
    os << " " << p.atype_;
    if (p.mtype_.has_value()) {
        os << " " << *p.mtype_;
    }
    if (p.closure_.isValid()) {
        os << " @" << p.closure_.get();
        if (p.updateClosure_) {
            os << "!";
        }
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const NativeCodeRet &r) {
    os << r.name_ << ": " << r.dtype_;
    if (r.closure_.isValid()) {
        os << " @" << r.closure_.get();
        if (r.returnClosure_) {
            os << "!";
        }
    }
    return os;
}

NativeCode NativeCode::fromFunc(const Func &func, const std::string &code,
                                const Ref<Target> &target) {
    FindSignatureTypes finder(func->params_, func->returns_);
    finder(func->body_);
    return NativeCode(func->name_, finder.params(), finder.returns(), code,
                      target);
}

} // namespace freetensor
