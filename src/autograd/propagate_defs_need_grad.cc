#include <analyze/find_stmt.h>
#include <autograd/propagate_defs_need_grad.h>

namespace freetensor {

void PropagateRequires::visit(const Load &op) {
    if (isFloat(op->dtype()) && curTarget_.isValid() &&
        affectedDefs_.count(def(op->var_)->id())) {
        affectedDefs_.insert(curTarget_);
        // No need to recurse deeper
    }
}

void PropagateRequires::visit(const Store &op) {
    if (buffer(op->var_)->atype() == AccessType::Cache) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateRequires::visit(const ReduceTo &op) {
    if (buffer(op->var_)->atype() == AccessType::Cache) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateRequires::visit(const VarDef &op) {
    if (requires_.count(op->name_) || provides_.count(op->name_)) {
        affectedDefs_.insert(op->id());
    }
    BaseClass::visit(op);
}

std::unordered_set<ID> PropagateRequires::propagateUntilConverge(
    const Stmt &op, const std::unordered_set<std::string> &_requires,
    const std::unordered_set<std::string> &provides) {
    for (auto &&name : _requires) {
        try {
            findStmt(op, [&](const Stmt &s) {
                return s->nodeType() == ASTNodeType::VarDef &&
                       s.as<VarDefNode>()->name_ == name;
            });
        } catch (const UnexpectedQueryResult &e) {
            throw InvalidAutoGrad("Input variable requesting for gradient `" +
                                  name + "` is not found or duplicated");
        }
    }
    for (auto &&name : provides) {
        try {
            findStmt(op, [&](const Stmt &s) {
                return s->nodeType() == ASTNodeType::VarDef &&
                       s.as<VarDefNode>()->name_ == name;
            });
        } catch (const UnexpectedQueryResult &e) {
            throw InvalidAutoGrad("Output variable providing gradient `" +
                                  name + "` is not found or duplicated");
        }
    }

    PropagateRequires propagator(_requires, provides);
    size_t affectCnt;
    do {
        affectCnt = propagator.affectedDefs().size();
        propagator(op);
    } while (propagator.affectedDefs().size() > affectCnt);
    return propagator.affectedDefs();
}

void PropagateProvides::visit(const Load &op) {
    if (isFloat(op->dtype()) && curTarget_.isValid() &&
        buffer(op->var_)->atype() == AccessType::Cache) {
        affectedDefs_.insert(def(op->var_)->id());
        // No need to recurse deeper
    }
}

void PropagateProvides::visit(const Store &op) {
    if (affectedDefs_.count(def(op->var_)->id())) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateProvides::visit(const ReduceTo &op) {
    if (affectedDefs_.count(def(op->var_)->id())) {
        curTarget_ = def(op->var_)->id();
        (*this)(op->expr_);
        // No need to recurse into indices
        curTarget_ = {};
    }
}

void PropagateProvides::visit(const VarDef &op) {
    if (requires_.count(op->name_) || provides_.count(op->name_)) {
        affectedDefs_.insert(op->id());
    }
    BaseClass::visit(op);
}

std::unordered_set<ID> PropagateProvides::propagateUntilConverge(
    const Stmt &op, const std::unordered_set<std::string> &_requires,
    const std::unordered_set<std::string> &provides) {
    PropagateProvides propagator(_requires, provides);
    size_t affectCnt;
    do {
        affectCnt = propagator.affectedDefs().size();
        propagator(op);
    } while (propagator.affectedDefs().size() > affectCnt);
    return propagator.affectedDefs();
}

} // namespace freetensor
