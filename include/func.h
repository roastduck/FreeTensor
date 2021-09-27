#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include <ast.h>
#include <frontend_utils.h>
#include <stmt.h>
#include <tensor.h>
#include <buffer.h>

namespace ir {


class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    std::unordered_map<std::string, Ref<Buffer>> buffers_;
    SubTree<StmtNode> body_;
    Ref<pybind11::object> src_;

    DEFINE_NODE_TRAIT(Func);

    ~FuncNode() {
#pragma omp critical
        { src_ = nullptr; }
    }
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               const std::unordered_map<std::string, Ref<Buffer>> &buffers,
               Tbody &&body, const pybind11::object &src) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->buffers_ = buffers;
    f->body_ = std::forward<Tbody>(body);
#pragma omp critical
    { f->src_ = Ref<pybind11::object>::make(src); }
    return f;
}
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               const std::unordered_map<std::string, Ref<Buffer>> &buffers,
               Tbody &&body, const Ref<pybind11::object> &src) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->buffers_ = buffers;
    f->body_ = std::forward<Tbody>(body);
    f->src_ = src;
    return f;
}

Func deepCopy(const Func &func);

} // namespace ir

#endif // FUNC_H
