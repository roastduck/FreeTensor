#ifndef FUNC_H
#define FUNC_H

#include <string>
#include <vector>

#include <ast.h>
#include <stmt.h>
#include <tensor.h>

namespace ir {

class FuncNode : public ASTNode {
  public:
    std::string name_;
    std::vector<std::string> params_;
    SubTree<StmtNode> body_;

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

enum class FuncArgType : int { Name, Literal };

class FuncArg {
    FuncArgType type_;
    std::string name_;
    Ref<TensorData> literal_;

  public:
    FuncArgType type() const { return type_; }

    const std::string &name() const {
        ASSERT(type_ == FuncArgType::Name);
        return name_;
    }

    const TensorData &literal() const {
        ASSERT(type_ == FuncArgType::Literal);
        return *literal_;
    }

    static FuncArg fromName(const std::string &name) {
        FuncArg ret;
        ret.type_ = FuncArgType::Name;
        ret.name_ = name;
        return ret;
    }

    static FuncArg fromLiteral(const Ref<TensorData> &literal) {
        FuncArg ret;
        ret.type_ = FuncArgType::Literal;
        ret.literal_ = literal;
        return ret;
    }
};

/**
 * Prepare for inlining the function: Strip away the function signature and
 * leave the function body
 *
 * 1. Remove I/O VarDef nodes
 * 2. Rename all the IDs and variables
 */
Stmt func2stmt(const Func &func, const std::vector<FuncArg> &args,
               const std::string &callSiteId);

} // namespace ir

#endif // FUNC_H
