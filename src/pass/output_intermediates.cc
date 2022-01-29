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

bool OutputIntermediates::isSingleVersion(const std::string &defId) const {
    return totLens_.at(defId)->nodeType() == ASTNodeType::IntConst &&
           totLens_.at(defId).as<IntConstNode>()->val_ == 1;
}

Stmt OutputIntermediates::visit(const Store &op) {
    auto oldStore = Mutator::visit(op);
    if (versions_.count(op) && !isSingleVersion(defs_.at(op->var_)->id())) {
        std::vector<Expr> newIndices(1, versions_.at(op));
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
    auto oldReduce = Mutator::visit(op);
    if (versions_.count(op) && !isSingleVersion(defs_.at(op->var_)->id())) {
        std::vector<Expr> newIndices(1, versions_.at(op));
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
            ASSERT(!defs_.count(_op->name_));
            defs_[_op->name_] = _op;
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            defs_.erase(_op->name_);

            tapeNames_[op->id()] = op->name_;
            if (op->buffer_->atype() != AccessType::InOut) {
                op->buffer_->setAtype(AccessType::Output);
            }
            return op;
        } else {
            ASSERT(!defs_.count(_op->name_));
            defs_[_op->name_] = _op;
            auto __op = Mutator::visit(_op);
            ASSERT(__op->nodeType() == ASTNodeType::VarDef);
            auto op = __op.as<VarDefNode>();
            defs_.erase(_op->name_);

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
        ASSERT(!defs_.count(_op->name_));
        defs_[_op->name_] = _op;
        auto ret = Mutator::visit(_op);
        defs_.erase(_op->name_);
        return ret;
    }
}

std::tuple<Stmt, std::unordered_map<std::string, std::string>,
           std::unordered_map<AST, Expr>, std::unordered_map<std::string, Expr>>
outputIntermediates(const Stmt &op,
                    const std::unordered_set<std::string> &intermediates) {
    auto [versions, totLens] = analyzeVersion(op, intermediates);
    OutputIntermediates mutator(versions, totLens);
    auto ret = mutator(op);
    return std::make_tuple(ret, mutator.tapeNames(), versions, totLens);
}

} // namespace ir
