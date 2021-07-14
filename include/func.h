#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include <ast.h>
#include <frontend_utils.h>
#include <stmt.h>
#include <tensor.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    SubTree<StmtNode> body_;
    pybind11::object src_;

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               Tbody &&body, const pybind11::object &src) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->body_ = std::forward<Tbody>(body);
    f->src_ = src;
    return f;
}

Func deepCopy(const Func &func);

} // namespace ir

#endif // FUNC_H
