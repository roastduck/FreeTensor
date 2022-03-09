#include <algorithm>

#include <analyze/all_uses.h>
#include <analyze/analyze_linear.h>
#include <analyze/as_dnf.h>
#include <analyze/check_all_defined.h>
#include <math/bounds.h>
#include <pass/z3_simplify.h>
#include <schedule/separate_tail.h>

namespace ir {

static bool noIntersect(const std::unordered_set<std::string> &set1,
                        const std::unordered_set<std::string> &set2) {
    for (auto &&x : set1) {
        if (set2.count(x)) {
            return false;
        }
    }
    return true;
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

void SeparateTail::genSeparation(
    const Expr &iterVar, const Expr &_cond,
    const std::unordered_set<std::string> &bodyAllWrites,
    const std::function<void(const Expr &)> &callback) {
    for (auto &&conj : asDNF(_cond)) {
        for (auto &&cond : conj) {
            if (!noIntersect(allReads(cond), bodyAllWrites)) {
                continue;
            }

            auto norm = linearComp(cond);
            if (!norm.isValid()) {
                continue;
            }

            auto [lin, type] = *norm;
            auto [lower, upper] = lin2bounds(
                lin, type == ASTNodeType::NE ? ASTNodeType::EQ : type, iterVar);
            if (lower.isValid()) {
                callback(lower->expr());
            }
            if (upper.isValid()) {
                callback(makeAdd(upper->expr(), makeIntConst(1)));
            }
        }
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

    auto bodyAllWrites = allWrites(op);
    auto iterVar = makeVar(op->iter_);
    ASTHashSet<Expr> sepSet;
    for (auto &&branch : ifList) {
        genSeparation(iterVar, branch->cond_, bodyAllWrites,
                      [&](const Expr &sep) { sepSet.insert(sep); });
    }

    std::vector<Expr> separations;
    separations.reserve(sepSet.size());
    for (auto &&item : sepSet) {
        separations.emplace_back(item);
    }
    std::function<Stmt(size_t, const For &)> dfs =
        [&separations, &dfs, this](size_t i, const For &old) -> Stmt {
        if (i == separations.size()) {
            return old;
        }
        auto &&sep = separations[i];
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
