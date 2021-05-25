#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <vector>

#include <ast.h>
#include <stmt.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    Stmt body_;

    DEFINE_NODE_TRAIT(Func);
};
typedef Ref<FuncNode> Func;
#define makeFunc(...) makeNode(Func, __VA_ARGS__)
template <class Tbody>
Func _makeFunc(const std::string &name, const std::vector<std::string> &params,
               Tbody &&body) {
    Func f = Func::make();
    f->name_ = name;
    f->params_ = params;
    f->body_ = std::forward<Tbody>(body);
    return f;
}

/**
 * Prepare for inlining the function: Strip away the function signature and
 * leave the function body
 *
 * 1. Remove I/O VarDef nodes
 * 2. Rename all the IDs and variables
 */
Stmt func2stmt(const Func &func, const std::vector<std::string> &args,
               const std::string &callSiteId);

} // namespace ir

#endif // FUNC_H
