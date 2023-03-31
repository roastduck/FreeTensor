#include <analyze/all_defs.h>
#include <analyze/deps.h>
#include <analyze/symbol_table.h>
#include <autograd/find_tape_or_recomp_stmts.h>

namespace freetensor {

namespace {

class ReplaceMarkVersionByFakeLoad : public SymbolTable<Mutator> {
    typedef SymbolTable<Mutator> BaseClass;

  public:
    std::unordered_set<ID> ids_;

    const auto &ids() const { return ids_; }

  protected:
    using BaseClass::visit;

    Stmt visit(const MarkVersion &op) override {
        auto &&tensor = buffer(op->var_)->tensor();
        ids_.emplace(op->id());
        return makeEval(
            makeLoad(op->var_,
                     std::vector<Expr>(
                         tensor->shape().size(),
                         makeIntrinsic("__any__", {}, DataType::Int32, false)),
                     tensor->dtype()),
            op->metadata(), op->id());
    }
};

class FindWrites : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    std::function<void(const Expr & /* expr */, const Stmt & /* write */,
                       const ID & /* defId */)>
        callback_;

  public:
    FindWrites(const auto &callback) : callback_(callback) {}

  protected:
    using BaseClass::visit;

    void visit(const Store &op) override {
        BaseClass::visit(op);
        callback_(op->expr_, op, def(op->var_)->id());
    }

    void visit(const ReduceTo &op) override {
        BaseClass::visit(op);
        callback_(op->expr_, op, def(op->var_)->id());
    }
};

bool isInIndices(const Expr &expr) {
    AST p = expr;
    while (p->isExpr()) {
        p = p->parentAST();
        switch (p->nodeType()) {
        case ASTNodeType::Load:
            for (auto &&idx : p.as<LoadNode>()->indices_) {
                if (expr == idx) {
                    return true;
                }
            }
        case ASTNodeType::Store:
            for (auto &&idx : p.as<StoreNode>()->indices_) {
                if (expr == idx) {
                    return true;
                }
            }
        case ASTNodeType::ReduceTo:
            for (auto &&idx : p.as<ReduceToNode>()->indices_) {
                if (expr == idx) {
                    return true;
                }
            }
        default:;
        }
    }
    return false;
}

} // Anonymous namespace

std::pair<std::unordered_map<ID, std::unordered_set<ID>>,
          std::unordered_map<ID, std::unordered_set<ID>>>
findTapeOrRecompStmts(
    const Stmt &op, const std::unordered_set<ID> &defsToTape,
    std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives) {
    std::unordered_map<ID, std::unordered_set<ID>> idsToTape, idsToRecomp;

    // - (RAW) If the variable is written by statement X, then read by statement
    // Y, and Y's gradient needs X, a version of it on X is needed
    // - (RAW) If the variable is written by statement X, where the value may
    // propagate to a MarkVersion node, a version of it on X is needed. In order
    // to find such statements, we replace MarkVersion by a fake Load node with
    // relaxed indices before we do a dependence analysis.
    ReplaceMarkVersionByFakeLoad replacer;
    auto fakeOp = replacer(op);
    FindDeps().type(DEP_RAW).filterLater([&](const auto &acc) -> bool {
        if (acc.op_->nodeType() != ASTNodeType::Load) {
            return false;
        }
        if (replacer.ids().count(acc.stmt_->id())) {
            // It is a MarkVersion
            return true;
        }
        Expr rootExpr = acc.op_.template as<ExprNode>();
        for (auto p = rootExpr->parentExpr(); p.isValid();
             p = p->parentExpr()) {
            rootExpr = p;
        }
        if (isInIndices(acc.op_.template as<ExprNode>())) {
            // Use in indices: always needed
            return true;
        }
        if (auto it = derivatives.find(StmtOrExprID{rootExpr, acc.stmt_->id()});
            it != derivatives.end()) {
            // Use in values: needed if the derivative needs it
            return it->second.reads().count(acc.def_->name_);
        }
        return false;
    })(fakeOp, [&](const Dependence &d) {
        ASSERT(d.earlier()->nodeType() != ASTNodeType::Load);
        if (defsToTape.count(d.defId())) {
            idsToTape[d.defId()].insert(d.earlier().as<StmtNode>()->id());
        } else {
            idsToRecomp[d.defId()].insert(d.earlier().as<StmtNode>()->id());
        }
    });

    // - (W) If the variable is written by statement X, and then overwritten
    // by statement Y, and if we need X's result to compute X's gradient (use
    // `y` for `y = f(x)`'s gradient), then we need a version of it on X.
    FindWrites{[&](const Expr &expr, const Stmt &write, const ID &defId) {
        if (auto it = derivatives.find(StmtOrExprID{expr, write->id()});
            it != derivatives.end() && it->second.usingStore()) {
            if (defsToTape.count(defId)) {
                idsToTape[defId].insert(write->id());
            } else {
                idsToRecomp[defId].insert(write->id());
            }
        }

        // Special. See Grad::visit(ReduceTo)
        if (write->nodeType() == ASTNodeType::ReduceTo &&
            (write.as<ReduceToNode>()->op_ == ReduceOp::Min ||
             write.as<ReduceToNode>()->op_ == ReduceOp::Max)) {
            if (defsToTape.count(defId)) {
                idsToTape[defId].insert(write->id());
            } else {
                idsToRecomp[defId].insert(write->id());
            }
        }
    }}(op);

    // - (Transitive RAW) If a statement needs to be computed, all previous
    // statements that produce values required by this statement also need to be
    // recomputed, unless they are taped.
    bool converged = false;
    do {
        converged = true;
        FindDeps()
            .type(DEP_RAW)
            .filterAccess([&](const auto &acc) -> bool {
                return !idsToTape.count(acc.def_->id()) &&
                       idsToRecomp.count(acc.def_->id());
            })
            .filterEarlier([&](const auto &acc) -> bool {
                return !idsToRecomp.at(acc.def_->id()).count(acc.stmt_->id());
            })
            .filterLater([&](const auto &acc) -> bool {
                if (acc.op_->nodeType() != ASTNodeType::Load) {
                    return false;
                }
                return idsToRecomp.at(acc.def_->id()).count(acc.stmt_->id());
            })(op, [&](const Dependence &d) {
                ASSERT(d.earlier()->nodeType() != ASTNodeType::Load);
                idsToRecomp[d.defId()].insert(d.earlier().as<StmtNode>()->id());
                converged = false;
            });
    } while (!converged);

    // InOut variables must be taped
    for (auto &&[id, name] : allDefs(op, {AccessType::InOut})) {
        if (!idsToTape.count(id)) {
            idsToTape[id] = {};
        }
    }

    return {idsToTape, idsToRecomp};
}

} // namespace freetensor
