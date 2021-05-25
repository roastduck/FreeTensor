#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <vector>

#include <ast.h>
#include <stmt.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::vector<std::string> params_;
    Stmt body_;

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::vector<std::string> &params, Tbody &&body) {
    Func f = Func::make();
    f->params_ = params;
    f->body_ = std::forward<Tbody>(body);
    return f;
}

} // namespace ir

#endif // FUNC_H
