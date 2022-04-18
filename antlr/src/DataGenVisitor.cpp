#include "DataGenVisitor.h"

std::string DataGenVisitor::genData(
        const std::shared_ptr<ASTNode> &op,
        const std::unordered_map<std::string, ExprType> &typeInfo) {
    typeInfo_ = &typeInfo;
    (*this)(op);
    return os.str();
}

void DataGenVisitor::visit(const ProgramNode *op) {
    os << "\n"
          ".section .data\n";
    Visitor::visit(op);
    for (auto &&item : defMap_) {  // deduplicate
        os << item.second;
    }
}

void DataGenVisitor::visit(const GlobalVarDefNode *op) {
    switch (op->type_) {
        case ExprType::Int: {
            auto init = op->init_.empty() ? "0" : op->init_;
            defMap_[op->name_] = op->name_ + ":\n\t.dword " + init + "\n";
            break;
        }
        default:
            throw std::runtime_error("Unsupported global type");
    }
}

