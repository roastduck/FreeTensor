#ifndef FREE_TENSOR_FUNC_H
#define FREE_TENSOR_FUNC_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <driver/array.h>
#include <stmt.h>
#include <tensor.h>

namespace freetensor {

struct FuncParam {
    std::string name_;
    Ref<Ref<Array>> closure_; /// Data bound to this parameter
    bool updateClosure_;      /// Accept user input even if there is a closure

    bool isInClosure() const { return closure_.isValid(); }

    FuncParam(const std::string &name, const Ref<Ref<Array>> &closure,
              bool updateClosure)
        : name_(name), closure_(closure), updateClosure_(updateClosure) {}
};

std::ostream &operator<<(std::ostream &os, const FuncParam &p);

struct FuncRet {
    std::string name_;
    DataType dtype_;
    Ref<Ref<Array>> closure_; /// Data bound to this return value
    bool returnClosure_;      /// Return even if there is a closure

    bool isInClosure() const { return closure_.isValid(); }

    FuncRet(const std::string &name, DataType dtype,
            const Ref<Ref<Array>> &closure, bool returnClosure)
        : name_(name), dtype_(dtype), closure_(closure),
          returnClosure_(returnClosure) {}
};

std::ostream &operator<<(std::ostream &os, const FuncRet &r);

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<FuncParam> params_;
    std::vector<FuncRet>
        returns_; // NOTE: multiple items in `returns_` may share the same name.
                  // In this case, one variable should be returned to multiple
                  // positions
    SubTree<StmtNode> body_ = ChildOf{this};

    bool isFunc() const override { return true; }

    void compHash() override { ASSERT(false); } // TODO

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
template <class Tbody, class Tparams, class Treturns, class Tclosure>
Func makeFunc(const std::string &name, Tparams &&params, Treturns &&returns,
              Tbody &&body) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = std::forward<Tparams>(params);
    f->returns_ = std::forward<Treturns>(returns);
    f->body_ = std::forward<Tbody>(body);
    return f;
}
template <class Tbody>
Func makeFunc(const std::string &name, const std::vector<FuncParam> &params,
              const std::vector<FuncRet> &returns, Tbody &&body) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->returns_ = returns;
    f->body_ = std::forward<Tbody>(body);
    return f;
}

Func deepCopy(const Func &func);

#define DEFINE_PASS_FOR_FUNC(pass)                                             \
    template <typename... T> Func pass(const Func &func, T &&...args) {        \
        return makeFunc(func->name_, func->params_, func->returns_,            \
                        pass(func->body_, std::forward<T>(args)...));          \
    }

} // namespace freetensor

#endif // FREE_TENSOR_FUNC_H
