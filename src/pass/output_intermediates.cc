#include <analyze/analyze_version.h>
#include <pass/output_intermediates.h>

namespace ir {

static MemType toGlobalMemType(MemType mtype) {
    switch (mtype) {
    case MemType::CPU:
        return MemType::CPU;
    case MemType::GPULocal:
    case MemType::GPUShared:
    case MemType::GPUGlobal:
        return MemType::GPUGlobal;
    default:
        ASSERT(false);
    }
}

bool OutputIntermediates::isSingleVersion(const ID &defId) const {
    return totLens_.at(defId)->nodeType() == ASTNodeType::IntConst &&
           totLens_.at(defId).as<IntConstNode>()->val_ == 1;
}

Stmt OutputIntermediates::visit(const Store &op) {
    auto oldStore = BaseClass::visit(op);
    if (versions_.count(op->id()) && !isSingleVersion(def(op->var_)->id())) {
        std::vector<Expr> newIndices(1, versions_.at(op->id()));
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        auto newStore = makeStore("", op->var_ + ".tape", std::move(newIndices),
                                  makeLoad(op->var_, op->indices_));
        return makeStmtSeq("", {oldStore, newStore});
    } else {
        return oldStore;
    }
}

Stmt OutputIntermediates::visit(const ReduceTo &op) {
    auto oldReduce = BaseClass::visit(op);
    if (versions_.count(op->id()) && !isSingleVersion(def(op->var_)->id())) {
        std::vector<Expr> newIndices(1, versions_.at(op->id()));
        newIndices.insert(newIndices.end(), op->indices_.begin(),
                          op->indices_.end());
        auto newStore = makeStore("", op->var_ + ".tape", std::move(newIndices),
                                  makeLoad(op->var_, op->indices_));
        return makeStmtSeq("", {oldReduce, newStore});
    } else {
        return oldReduce;
    }
}

Stmt OutputIntermediates::visit(const VarDef &_op) {
    if (totLens_.count(_op->id())) {
        // FIXME: What if the scopeLen_ is a loop-variant temporary?
        if (isSingleVersion(_op->id())) {
            // No need to create a new VarDef
            auto __op = BaseClass::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();

            tapeNames_[op->id()] = op->name_;
            if (op->buffer_->atype() != AccessType::InOut) {
                op->buffer_->setAtype(AccessType::Output);
            }
            return op;
        } else {
            auto __op = BaseClass::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();

            auto tapeName = tapeNames_[op->id()] = op->name_ + ".tape";
            auto tensor = op->buffer_->tensor();
            tensor.shape().insert(tensor.shape().begin(),
                                  totLens_.at(op->id()));
            return makeVarDef("", tapeName,
                              Buffer(std::move(tensor), AccessType::Output,
                                     toGlobalMemType(op->buffer_->mtype())),
                              nullptr, op, false);
        }
    } else {
        return BaseClass::visit(_op);
    }
}

std::tuple<Stmt, std::unordered_map<ID, std::string>,
           std::unordered_map<ID, Expr>, std::unordered_map<ID, Expr>>
outputIntermediates(const Stmt &op,
                    const std::unordered_set<ID> &intermediates) {
    auto [versions, totLens] = analyzeVersion(op, intermediates);
    OutputIntermediates mutator(versions, totLens);
    auto ret = mutator(op);
    return std::make_tuple(ret, mutator.tapeNames(), versions, totLens);
}

} // namespace ir
