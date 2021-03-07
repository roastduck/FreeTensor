#include <algorithm>

#include <analyze/check_all_defined.h>
#include <pass/seperate_tail.h>
#include <pass/z3_simplify.h>

namespace ir {

static ASTNodeType reverseCmp(ASTNodeType type) {
    switch (type) {
    case ASTNodeType::LT:
        return ASTNodeType::GT;
    case ASTNodeType::LE:
        return ASTNodeType::GE;
    case ASTNodeType::GT:
        return ASTNodeType::LT;
    case ASTNodeType::GE:
        return ASTNodeType::LE;
    default:
        ASSERT(false);
    }
}

void FindAllIfs::visit(const If &op) {
    Visitor::visit(op);
    results_.insert(op->id());
}

Stmt AppendIDs::visitStmt(const Stmt &op,
                        const std::function<Stmt(const Stmt &)> &visitNode) {
    auto ret = Mutator::visitStmt(op, visitNode);
    ret->setId(op->id() + suffix_);
    return ret;
}

void SeperateTail::genSeperation(
    uint64_t iterHash, const Expr &cond,
    const std::function<void(const Expr &)> &callback) {
    auto type = cond->nodeType();

    Expr norm;
    switch (type) {
    case ASTNodeType::LAnd:
        genSeperation(iterHash, cond.as<LAndNode>()->lhs_, callback);
        genSeperation(iterHash, cond.as<LAndNode>()->rhs_, callback);
        return;
    case ASTNodeType::LOr:
        genSeperation(iterHash, cond.as<LOrNode>()->lhs_, callback);
        genSeperation(iterHash, cond.as<LOrNode>()->rhs_, callback);
        return;
    case ASTNodeType::LNot:
        genSeperation(iterHash, cond.as<LNotNode>()->expr_, callback);
        return;
    case ASTNodeType::LT:
        norm = makeSub(cond.as<LTNode>()->lhs_, cond.as<LTNode>()->rhs_);
        break;
    case ASTNodeType::LE:
        norm = makeSub(cond.as<LENode>()->lhs_, cond.as<LENode>()->rhs_);
        break;
    case ASTNodeType::GT:
        norm = makeSub(cond.as<GTNode>()->lhs_, cond.as<GTNode>()->rhs_);
        break;
    case ASTNodeType::GE:
        norm = makeSub(cond.as<GENode>()->lhs_, cond.as<GENode>()->rhs_);
        break;
    default:
        return;
    }
    ASSERT(norm.isValid());
    analyzeLinear_(norm);
    if (!analyzeLinear_.result().count(norm)) {
        return;
    }
    LinearExpr lin = analyzeLinear_.result().at(norm);

    if (!lin.coeff_.count(iterHash)) {
        return;
    }
    auto selfK = lin.coeff_.at(iterHash).k_;
    if (selfK < 0) {
        type = reverseCmp(type);
        selfK *= -1;
        lin.bias_ *= -1;
        for (auto &item : lin.coeff_) {
            item.second.k_ *= -1;
        }
    }

    Expr seperation = makeIntConst(-lin.bias_);
    for (auto &&item : lin.coeff_) {
        if (item.first != iterHash) {
            seperation =
                makeAdd(seperation,
                        makeMul(makeIntConst(-item.second.k_), item.second.a_));
        }
    }
    switch (type) {
    case ASTNodeType::LT:
    case ASTNodeType::GE:
        seperation = makeCeilDiv(seperation, makeIntConst(selfK));
        break;
    case ASTNodeType::LE:
    case ASTNodeType::GT:
        seperation = makeAdd(makeFloorDiv(seperation, makeIntConst(selfK)),
                            makeIntConst(1));
        break;
    default:
        ASSERT(false);
    }
    callback(seperation);
}

Stmt SeperateTail::visit(const If &op) {
    if (candidates_.count(op->id())) {
        for (auto &item : ifStack_) {
            item.emplace_back(op); // Use the old one
        }
    }
    return Mutator::visit(op);
}

Stmt SeperateTail::visit(const For &_op) {
    def_.insert(_op->iter_);
    ifStack_.emplace_back();

    auto __op = Mutator::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    std::vector<If> ifList;
    for (auto &&branch : ifStack_.back()) {
        if (checkAllDefined(def_, branch->cond_)) {
            ifList.emplace_back(branch);
        }
    }

    def_.erase(_op->iter_);
    ifStack_.pop_back();

    if (!op->parallel_.empty()) {
        return op;
    }

    auto iterHash = getHash(makeVar(op->iter_));
    std::vector<Expr> seperations;
    for (auto &&branch : ifList) {
        genSeperation(iterHash, branch->cond_,
                    [&](const Expr &sep) { seperations.emplace_back(sep); });
    }

    std::function<Stmt(size_t, const For &)> dfs =
        [&seperations, &dfs, this](size_t i, const For &old) -> Stmt {
        if (i == seperations.size()) {
            return old;
        }
        auto &&sep = seperations[i];
        auto front =
            makeFor(old->id(), old->iter_, old->begin_, makeMin(old->end_, sep),
                    old->parallel_, old->body_);
        auto back = makeFor(old->id(), old->iter_, makeMax(old->begin_, sep),
                            old->end_, old->parallel_, old->body_);
        front = dfs(i + 1, AppendIDs(".front")(front));
        back = dfs(i + 1, AppendIDs(".back")(back));
        auto seperated = makeStmtSeq("", {front, back});
        auto ret = makeIf(
            "", makeLAnd(makeGE(sep, old->begin_), makeLT(sep, old->end_)),
            seperated, old);
        nextCandidates_.insert(ret->id());
        return ret;
    };
    return dfs(0, op);
}

Stmt SeperateTail::visit(const VarDef &op) {
    def_.insert(op->name_);
    auto ret = Mutator::visit(op);
    def_.erase(op->name_);
    return ret;
}

Stmt seperateTail(const Stmt &_op) {
    auto op = _op;

    FindAllIfs finder;
    finder(op);
    auto candidates = finder.results();

    while (!candidates.empty()) {
        SeperateTail mutator(candidates);
        op = mutator(op);
        op = z3Simplify(op); // Although Z3 may be slow, if we don't use Z3
                            // here, there will be too many redundant branches,
                            // which will make each pass even slower
        candidates = mutator.nextCandidates();
    }

    return op;
}

} // namespace ir

