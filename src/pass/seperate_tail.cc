#include <algorithm>

#include <analyze/check_all_defined.h>
#include <analyze/hash.h>
#include <analyze/normalize.h>
#include <pass/seperate_tail.h>
#include <pass/simplify.h>

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

static Expr makeCeilDiv(const Expr &lhs, const Expr &rhs) {
    return makeAdd(makeDiv(makeSub(lhs, makeIntConst(1)), rhs),
                   makeIntConst(1));
}

Stmt SeperateTail::visit(const If &op) {
    for (auto &item : ifStack_) {
        item.emplace_back(op); // Use the old one
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
        auto type = branch->cond_->nodeType();

        Expr norm;
        switch (type) {
        case ASTNodeType::LT:
            norm = branch->cond_.as<LTNode>()->info_norm_form_;
            break;
        case ASTNodeType::LE:
            norm = branch->cond_.as<LENode>()->info_norm_form_;
            break;
        case ASTNodeType::GT:
            norm = branch->cond_.as<GTNode>()->info_norm_form_;
            break;
        case ASTNodeType::GE:
            norm = branch->cond_.as<GENode>()->info_norm_form_;
            break;
        default:
            continue;
        }
        ASSERT(norm.isValid());
        if (!linear_.count(norm)) {
            continue;
        }
        LinearExpr lin = linear_.at(norm);

        if (!lin.coeff_.count(iterHash)) {
            continue;
        }
        auto selfK = lin.coeff_.at(iterHash).k;
        if (selfK < 0) {
            type = reverseCmp(type);
            selfK *= -1;
            lin.bias_ *= -1;
            for (auto &item : lin.coeff_) {
                item.second.k *= -1;
            }
        }

        Expr seperation = makeIntConst(-lin.bias_);
        for (auto &&item : lin.coeff_) {
            if (item.first != iterHash) {
                seperation =
                    makeAdd(seperation, makeMul(makeIntConst(-item.second.k),
                                                item.second.a));
            }
        }
        switch (type) {
        case ASTNodeType::LT:
        case ASTNodeType::GE:
            seperation = makeCeilDiv(seperation, makeIntConst(selfK));
            break;
        case ASTNodeType::LE:
        case ASTNodeType::GT:
            seperation = makeAdd(makeDiv(seperation, makeIntConst(selfK)),
                                 makeIntConst(1));
            break;
        default:
            ASSERT(false);
        }
        seperations.emplace_back(std::move(seperation));
    }

    std::function<Stmt(size_t, const For &)> dfs =
        [&seperations, &dfs](size_t i, const For &old) -> Stmt {
        if (i == seperations.size()) {
            return old;
        }
        auto &&sep = seperations[i];
        auto front =
            makeFor("", old->iter_, old->begin_, makeMin(old->end_, sep),
                    old->parallel_, old->body_);
        auto back = makeFor("", old->iter_, makeMax(old->begin_, sep),
                            old->end_, old->parallel_, old->body_);
        front = dfs(i + 1, front);
        back = dfs(i + 1, back);
        auto seperated = makeStmtSeq("", {front, back});
        return makeIf(
            "", makeLAnd(makeGE(sep, old->begin_), makeLT(sep, old->end_)),
            seperated, old);
    };
    return dfs(0, op);
}

Stmt SeperateTail::visit(const VarDef &op) {
    def_.insert(op->name_);
    auto ret = Mutator::visit(op);
    def_.erase(op->name_);
    return ret;
}

void CountNestedFor::visit(const For &op) {
    curNested_++;
    maxNested_ = std::max(maxNested_, curNested_);
    Visitor::visit(op);
    curNested_--;
}

Stmt seperateTail(const Stmt &_op) {
    Stmt op = normalize(_op);

    CountNestedFor counter;
    counter(op);
    int maxNested = counter.maxNested();

    for (int i = 0; i < maxNested; i++) {
        auto hash = getHashMap(op);

        AnalyzeLinear analyzeLinear(hash);
        analyzeLinear(op);
        auto &&linear = analyzeLinear.result();

        SeperateTail mutator(linear);
        op = mutator(op);

        op = simplifyPass(op);
    }
    return op;
}

} // namespace ir

