#include <algorithm>

#include <analyze/check_all_defined.h>
#include <pass/z3_simplify.h>
#include <schedule/separate_tail.h>

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

Stmt AppendIDs::visitStmt(const Stmt &op) {
    auto ret = Mutator::visitStmt(op);
    ret->setId(op->id().strId() + suffix_);
    return ret;
}

void SeparateTail::genSeperation(
    const Expr &iterVar, const Expr &cond,
    const std::function<void(const Expr &)> &callback) {
    auto type = cond->nodeType();

    Expr norm;
    switch (type) {
    case ASTNodeType::LAnd:
        genSeperation(iterVar, cond.as<LAndNode>()->lhs_, callback);
        genSeperation(iterVar, cond.as<LAndNode>()->rhs_, callback);
        return;
    case ASTNodeType::LOr:
        genSeperation(iterVar, cond.as<LOrNode>()->lhs_, callback);
        genSeperation(iterVar, cond.as<LOrNode>()->rhs_, callback);
        return;
    case ASTNodeType::LNot:
        genSeperation(iterVar, cond.as<LNotNode>()->expr_, callback);
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
    case ASTNodeType::EQ:
        norm = makeSub(cond.as<EQNode>()->lhs_, cond.as<EQNode>()->rhs_);
        break;
    case ASTNodeType::NE:
        norm = makeSub(cond.as<NENode>()->lhs_, cond.as<NENode>()->rhs_);
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

    auto it =
        std::find_if(lin.coeff_.begin(), lin.coeff_.end(),
                     [&iterVar](const decltype(lin.coeff_)::value_type &kx) {
                         return HashComparator()(kx.a_, iterVar);
                     });
    if (it == lin.coeff_.end()) {
        return;
    }
    auto selfK = it->k_;
    if (selfK < 0) {
        type = reverseCmp(type);
        selfK *= -1;
        lin.bias_ *= -1;
        for (auto &item : lin.coeff_) {
            item.k_ *= -1;
        }
    }

    Expr seperation = makeIntConst(-lin.bias_);
    for (auto &&item : lin.coeff_) {
        if (!HashComparator()(item.a_, iterVar)) {
            seperation =
                makeAdd(seperation, makeMul(makeIntConst(-item.k_), item.a_));
        }
    }
    switch (type) {
    case ASTNodeType::LT:
    case ASTNodeType::GE:
        callback(makeCeilDiv(seperation, makeIntConst(selfK)));
        break;
    case ASTNodeType::LE:
    case ASTNodeType::GT:
        callback(makeAdd(makeFloorDiv(seperation, makeIntConst(selfK)),
                         makeIntConst(1)));
        break;
    case ASTNodeType::EQ:
    case ASTNodeType::NE:
        callback(makeCeilDiv(seperation, makeIntConst(selfK)));
        callback(makeAdd(makeFloorDiv(seperation, makeIntConst(selfK)),
                         makeIntConst(1)));
        break;
    default:
        ASSERT(false);
    }
}

Stmt SeparateTail::visit(const If &op) {
    if (candidates_.count(op->id())) {
        for (auto &item : ifStack_) {
            item.emplace_back(op); // Use the old one
        }
    }
    return BaseClass::visit(op);
}

Stmt SeparateTail::visit(const For &_op) {
    ifStack_.emplace_back();
    hasVarDefStack_.emplace_back(false);

    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::For);
    auto op = __op.as<ForNode>();
    bool hasVarDef = hasVarDefStack_.back();

    pushFor(op);
    std::vector<If> ifList;
    for (auto &&branch : ifStack_.back()) {
        if (checkAllDefined(names(), branch->cond_)) {
            ifList.emplace_back(branch);
        }
    }
    popFor(op);

    ifStack_.pop_back();
    hasVarDefStack_.pop_back();

    if (noDuplicateVarDefs_ && hasVarDef) {
        return op;
    }

    if (!op->property_.parallel_.empty()) {
        return op;
    }

    auto iterVar = makeVar(op->iter_);
    ASTHashSet<Expr> sepSet;
    for (auto &&branch : ifList) {
        genSeperation(iterVar, branch->cond_,
                      [&](const Expr &sep) { sepSet.insert(sep); });
    }

    std::vector<Expr> seperations;
    seperations.reserve(sepSet.size());
    for (auto &&item : sepSet) {
        seperations.emplace_back(item);
    }
    std::function<Stmt(size_t, const For &)> dfs =
        [&seperations, &dfs, this](size_t i, const For &old) -> Stmt {
        if (i == seperations.size()) {
            return old;
        }
        auto &&sep = seperations[i];
        auto front =
            makeFor(old->id(), old->iter_, old->begin_, sep, old->step_,
                    makeFloorDiv(makeSub(sep, old->begin_), old->step_),
                    old->property_, old->body_);
        auto back = makeFor(
            old->id(), old->iter_,
            makeAdd(makeMul(makeCeilDiv(makeSub(sep, old->begin_), old->step_),
                            old->step_),
                    old->begin_),
            old->end_, old->step_,
            makeFloorDiv(makeSub(old->end_, sep), old->step_), old->property_,
            old->body_);
        front = dfs(i + 1, AppendIDs(".front")(front).as<ForNode>());
        back = dfs(i + 1, AppendIDs(".back")(back).as<ForNode>());
        auto separated = makeStmtSeq("", {front, back});
        auto ret = makeIf(
            "", makeLAnd(makeGE(sep, old->begin_), makeLE(sep, old->end_)),
            separated, dfs(i + 1, old));
        nextCandidates_.insert(ret->id());
        return ret;
    };
    return dfs(0, op);
}

Stmt SeparateTail::visit(const VarDef &op) {
    auto ret = BaseClass::visit(op);
    for (auto it = hasVarDefStack_.begin(); it != hasVarDefStack_.end(); it++) {
        *it = true;
    }
    return ret;
}

Stmt separateTail(const Stmt &_ast, bool noDuplicateVarDefs) {
    auto ast = _ast;

    FindAllIfs finder;
    finder(ast);
    auto candidates = finder.results();

    while (!candidates.empty()) {
        SeparateTail mutator(noDuplicateVarDefs, candidates);
        ast = mutator(ast);
        ast =
            z3Simplify(ast); // Although Z3 may be slow, if we don't use Z3
                             // here, there will be too many redundant branches,
                             // which will make each pass even slower
        candidates = mutator.nextCandidates();
    }

    return ast;
}

} // namespace ir
