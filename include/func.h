#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <utility>
#include <vector>

#include <ast.h>
#include <buffer.h>
#include <driver/array.h>
#include <frontend_utils.h>
#include <stmt.h>
#include <tensor.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    std::vector<std::pair<std::string, DataType>> returns_;
    SubTree<StmtNode> body_;

    // Some parameters and/or return values can be enclosed in `closure_`. These
    // values will be automatically set and collect in `Driver`. They are still
    // recorded in `params_` and/or `returns_`, but we do not have to specify
    // them to `Driver`
    std::unordered_map<std::string, Ref<Ref<Array>>> closure_;

    bool isFunc() const override { return true; }

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody, class Tparams, class Treturns, class Tclosure>
Func _makeFunc(const std::string &name, Tparams &&params, Treturns &&returns,
               Tbody &&body, Tclosure &&closure) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = std::forward<Tparams>(params);
    f->returns_ = std::forward<Treturns>(returns);
    f->body_ = std::forward<Tbody>(body);
    f->closure_ = std::forward<Tclosure>(closure);
    return f;
}
template <class Tbody>
Func _makeFunc(
    const std::string &name, const std::vector<std::string> &params,
    const std::vector<std::pair<std::string, DataType>> &returns, Tbody &&body,
    const std::unordered_map<std::string, Ref<Ref<Array>>> &closure) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->returns_ = returns;
    f->body_ = std::forward<Tbody>(body);
    f->closure_ = closure;
    return f;
}

Func deepCopy(const Func &func);

#define DEFINE_PASS_FOR_FUNC(pass)                                             \
    template <typename... T> Func pass(const Func &func, T &&...args) {        \
        return makeFunc(func->name_, func->params_, func->returns_,            \
                        pass(func->body_, std::forward<T>(args)...),           \
                        func->closure_);                                       \
    }

} // namespace ir

#endif // FUNC_H
