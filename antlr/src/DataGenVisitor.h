#ifndef DATA_GEN_VISITOR_H_
#define DATA_GEN_VISITOR_H_

#include <string>
#include <sstream>
#include <unordered_map>

#include "Visitor.h"

class DataGenVisitor : public Visitor {
public:
    std::string genData(
            const std::shared_ptr<ASTNode> &op,
            const std::unordered_map<std::string, ExprType> &typeInfo);

protected:
    void visit(const ProgramNode *op) override;
    void visit(const GlobalVarDefNode *op) override;

private:
    std::ostringstream os;
    std::unordered_map<std::string, std::string> defMap_;
    const std::unordered_map<std::string, ExprType> *typeInfo_;
};

#endif  // DATA_GEN_VISITOR_H_

