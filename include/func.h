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

enum class FuncArgIdxType : int { Single, Slice };

class FuncArgIdx {
    FuncArgIdxType type_;
    Expr start_, stop_;

  public:
    FuncArgIdxType type() const { return type_; }

    const Expr &single() const {
        ASSERT(type_ == FuncArgIdxType::Single);
        return start_;
    }

    const Expr &start() const {
        ASSERT(type_ == FuncArgIdxType::Slice);
        return start_;
    }

    const Expr &stop() const {
        ASSERT(type_ == FuncArgIdxType::Slice);
        return stop_;
    }

    static FuncArgIdx fromSingle(const Expr &single) {
        FuncArgIdx ret;
        ret.type_ = FuncArgIdxType::Single;
        ret.start_ = single;
        return ret;
    }

    static FuncArgIdx fromSlice(const Expr &start, const Expr &stop) {
        FuncArgIdx ret;
        ret.type_ = FuncArgIdxType::Slice;
        ret.start_ = start;
        ret.stop_ = stop;
        return ret;
    }
};

enum class FuncArgType : int { Var, Literal };

class FuncArg {
    FuncArgType type_;

    // For Var
    std::string name_;
    std::vector<FuncArgIdx> indices_;

    // For Literal
    Ref<TensorData> literal_;

  public:
    FuncArgType type() const { return type_; }

    const std::string &name() const {
        ASSERT(type_ == FuncArgType::Var);
        return name_;
    }

    const std::vector<FuncArgIdx> &indices() const {
        ASSERT(type_ == FuncArgType::Var);
        return indices_;
    }

    const TensorData &literal() const {
        ASSERT(type_ == FuncArgType::Literal);
        return *literal_;
    }

    static FuncArg fromVar(const std::string &name,
                           const std::vector<FuncArgIdx> &indices) {
        FuncArg ret;
        ret.type_ = FuncArgType::Var;
        ret.name_ = name;
        ret.indices_ = indices;
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
