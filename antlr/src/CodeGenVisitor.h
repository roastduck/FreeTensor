#ifndef CODE_GEN_VISITOR_H_
#define CODE_GEN_VISITOR_H_

#include <string>
#include <sstream>
#include <unordered_map>

#include "Visitor.h"

class CodeGenVisitor : public Visitor {
public:
    std::string genCode(
            const std::shared_ptr<ASTNode> &op,
            const std::unordered_map<std::string, int> &varMap,
            const std::unordered_map<std::string, ExprType> &typeInfo);

protected:
    virtual void visit(const ProgramNode *op) override;
    virtual void visit(const FunctionNode *op) override;
    virtual void visit(const VarNode *op) override;
    virtual void visit(const AssignNode *op) override;
    virtual void visit(const InvokeNode *op) override;
    virtual void visit(const IfThenElseNode *op) override;
    virtual void visit(const WhileNode *op) override;
    virtual void visit(const DoWhileNode *op) override;
    virtual void visit(const ForNode *op) override;
    virtual void visit(const ReturnNode *op) override;
    virtual void visit(const BreakNode *op) override;
    virtual void visit(const ContinueNode *op) override;
    virtual void visit(const IntegerNode *op) override;
    virtual void visit(const CallNode *op) override;
    virtual void visit(const CastNode *op) override;
    virtual void visit(const AddNode *op) override;
    virtual void visit(const SubNode *op) override;
    virtual void visit(const MulNode *op) override;
    virtual void visit(const DivNode *op) override;
    virtual void visit(const ModNode *op) override;
    virtual void visit(const BAndNode *op) override;
    virtual void visit(const BOrNode *op) override;
    virtual void visit(const BXorNode *op) override;
    virtual void visit(const SLLNode *op) override;
    virtual void visit(const SRANode *op) override;
    virtual void visit(const LNotNode *op) override;
    virtual void visit(const LTNode *op) override;
    virtual void visit(const LENode *op) override;
    virtual void visit(const GTNode *op) override;
    virtual void visit(const GENode *op) override;
    virtual void visit(const EQNode *op) override;
    virtual void visit(const NENode *op) override;
    virtual void visit(const LAndNode *op) override;
    virtual void visit(const LOrNode *op) override;
    virtual void visit(const SelectNode *op) override;

private:
    void stmtPrelude();
    std::string getFullname(const std::string &name) const;

    const char *push =      "addi sp, sp, -8\n"
                            "sd a0, (sp)\n";
    const char *pop2 =      "ld t0, 8(sp)\n"
                            "ld t1, (sp)\n"
                            "addi sp, sp, 16\n";
    const char *puttop =    "sd a0, (sp)\n";

    std::ostringstream os;
    std::string curFunc_;
    const std::unordered_map<std::string, int> *varMap_;
    const std::unordered_map<std::string, ExprType> *typeInfo_;
    int jumpCnt_ = 0;
    int retTarget_, breakTarget_, continueTarget_;
    int curFuncNVar_;
};

#endif  // CODE_GEN_VISITOR_H_
