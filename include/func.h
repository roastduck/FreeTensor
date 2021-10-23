#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <utility>
#include <vector>

#include <ast.h>
#include <buffer.h>
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

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               const std::vector<std::pair<std::string, DataType>> &returns,
               Tbody &&body) {
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

} // namespace ir

#endif // FUNC_H
