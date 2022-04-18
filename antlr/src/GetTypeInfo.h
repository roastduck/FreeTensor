#ifndef GET_TYPE_INFO_H_
#define GET_TYPE_INFO_H_

#include <unordered_map>

#include "Visitor.h"

class GetTypeInfo : public Visitor {
public:
    std::unordered_map<std::string, ExprType> get(const std::shared_ptr<ASTNode> &op);

protected:
    virtual void visit(const FunctionNode *op) override;
    virtual void visit(const GlobalVarDefNode *op) override;
    virtual void visit(const VarDefNode *op) override;

private:
    std::unordered_map<std::string, ExprType> types_;
};

#endif  // GET_TYPE_INFO_H_
