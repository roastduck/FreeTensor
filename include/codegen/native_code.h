#ifndef FREE_TENSOR_NATIVE_CODE_H
#define FREE_TENSOR_NATIVE_CODE_H

#include <optional>
#include <string>

#include <driver/array.h>
#include <driver/target.h>
#include <func.h>
#include <type/access_type.h>
#include <type/data_type.h>
#include <type/mem_type.h>

namespace freetensor {

/**
 * Declare a parameter of a `NativeCode`
 *
 * A `NativeCodeParam` contains all information from a `FuncParam`, plus the
 * type information from the `VarDef` nodes of the function body (so `Driver`
 * can run without the function body AST).
 *
 * A parameter can be safely ignored by the program. In this case, `atype_` is
 * set to `AccessType::Bypass`, while `dtype_` and `mtype_` is unset. NOTE:
 * Since we allow removing a parameter, please be aware of potential bugs that a
 * newly introduced parameter takes the same name with the removed one.
 * Currently we only introduce new parameters in autograd, with ".grad" and
 * ".tape" suffix in names, so it's OK.
 */
struct NativeCodeParam {
    std::string name_;
    std::optional<DataType> dtype_; /// Null if atype_ == Bypass
    AccessType atype_;
    std::optional<MemType> mtype_; /// Null if atype_ == Bypass
    Ref<Ref<Array>> closure_;      /// Data bound to this parameter
    bool updateClosure_; /// Accept user input even if there is a closure

    bool isInClosure() const { return closure_.isValid(); }

    NativeCodeParam(const std::string &name,
                    const std::optional<DataType> &dtype,
                    const AccessType &atype,
                    const std::optional<MemType> &mtype,
                    const Ref<Ref<Array>> &closure, bool updateClosure)
        : name_(name), dtype_(dtype), atype_(atype), mtype_(mtype),
          closure_(closure), updateClosure_(updateClosure) {}
};

std::ostream &operator<<(std::ostream &os, const NativeCodeParam &p);

/**
 * Declare a return value of a `NativeCode`
 *
 * Currently this class contains the same information with a `FuncRet`
 */
struct NativeCodeRet {
    std::string name_;
    DataType dtype_;
    Ref<Ref<Array>> closure_; /// Data bound to this return value
    bool returnClosure_;      /// Return even if there is a closure

    bool isInClosure() const { return closure_.isValid(); }

    NativeCodeRet(const std::string &name, const DataType &dtype,
                  const Ref<Ref<Array>> &closure, bool returnClosure)
        : name_(name), dtype_(dtype), closure_(closure),
          returnClosure_(returnClosure) {}
};

std::ostream &operator<<(std::ostream &os, const NativeCodeRet &r);

/**
 * Generated native code with metadata
 */
class NativeCode {
    std::string name_;
    std::vector<NativeCodeParam> params_;
    std::vector<NativeCodeRet>
        returns_; // NOTE: multiple items in `returns_` may share the same name.
                  // In this case, one variable should be returned to multiple
                  // positions
    std::string code_;
    std::string entry_; /// Name of the function to be called
    Ref<Target> target_;

  public:
    NativeCode() {} // Uninitialized
    NativeCode(const std::string &name,
               const std::vector<NativeCodeParam> &params,
               const std::vector<NativeCodeRet> &returns,
               const std::string &code, const std::string &entry,
               const Ref<Target> &target)
        : name_(name), params_(params), returns_(returns), code_(code),
          entry_(entry), target_(target) {}

    static NativeCode fromFunc(const Func &func, const std::string &code,
                               const std::string &entry,
                               const Ref<Target> &target);

    const auto &name() const { return name_; }
    const auto &params() const { return params_; }
    const auto &returns() const { return returns_; }
    const auto &code() const { return code_; }
    const auto &entry() const { return entry_; }
    const auto &target() const { return target_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_NATIVE_CODE_H
