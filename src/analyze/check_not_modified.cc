#include <unordered_map>

#include <analyze/all_uses.h>
#include <analyze/check_not_modified.h>
#include <analyze/deps.h>
#include <analyze/find_stmt.h>
#include <pass/flatten_stmt_seq.h>

namespace freetensor {

static std::unordered_map<std::string, ID>
usedDefsAt(const Stmt &ast, const ID &pos,
           const std::unordered_set<std::string> &names) {
    CheckNameToDefMapping checker(pos, names);
    checker(ast);
    return checker.name2def();
}

void CheckNameToDefMapping::visitStmt(const Stmt &stmt) {
    BaseClass::visitStmt(stmt);
    if (stmt->id() == pos_) {
        for (auto &&name : names_) {
            if (hasDef(name)) {
                name2def_[name] = def(name)->id();
            } else if (hasLoop(name)) {
                name2def_[name] = loop(name)->id();
            }
        }
    }
}

static std::unordered_map<std::string, ID>
usedScopesAt(const Stmt &ast, const ID &pos,
             const std::unordered_set<std::string> &reads) {
    CheckReadToParallelScopeMapping checker(pos, reads);
    checker(ast);
    return checker.read2scope();
}

void CheckReadToParallelScopeMapping::visitStmt(const Stmt &stmt) {
    BaseClass::visitStmt(stmt);
    if (stmt->id() == pos_) {
        for (auto &&read : reads_) {
            switch (buffer(read)->mtype()) {
            case MemType::GPULocal:
            case MemType::GPUWarp:
                if (auto &&scope = stmt->parentStmtByFilter([](const Stmt &p) {
                        return p->nodeType() == ASTNodeType::For &&
                               std::holds_alternative<CUDAScope>(
                                   p.as<ForNode>()->property_->parallel_);
                    });
                    scope.isValid()) {
                    read2scope_[read] = scope->id();
                }
                break;
            case MemType::GPUShared:
                if (auto &&scope = stmt->parentStmtByFilter([](const Stmt &p) {
                        return p->nodeType() == ASTNodeType::For &&
                               std::holds_alternative<CUDAScope>(
                                   p.as<ForNode>()->property_->parallel_) &&
                               std::get<CUDAScope>(
                                   p.as<ForNode>()->property_->parallel_)
                                       .level_ == CUDAScope::Block;
                    });
                    scope.isValid()) {
                    read2scope_[read] = scope->id();
                }
                break;
            default:; // do nothing
            }
        }
    }
}

Stmt InsertTmpEval::visitStmt(const Stmt &_op) {
    auto op = Mutator::visitStmt(_op);
    auto ret = op;
    if (op->id() == s0_) {
        auto eval = makeEval(s0Expr_);
        s0Eval_ = eval->id();
        ret = s0Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq({eval, ret})
                  : makeStmtSeq({ret, eval});
    }
    if (op->id() == s1_) {
        auto eval = makeEval(s1Expr_);
        s1Eval_ = eval->id();
        ret = s1Side_ == CheckNotModifiedSide::Before
                  ? makeStmtSeq({eval, ret})
                  : makeStmtSeq({ret, eval});
    }
    return ret;
}

struct ModifiedException {};

bool checkNotModified(const Stmt &op, const Expr &s0Expr, const Expr &s1Expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1) {
    auto names = allUses(s0Expr); // uses of s1 should be the same
    if (names.empty()) {
        return true;
    }
    if (usedDefsAt(op, s0, names) != usedDefsAt(op, s1, names)) {
        return false;
    }

    auto reads = allReads(s0Expr); // uses of s1 should be the same
    if (reads.empty()) {
        return true; // early exit: impossible to be written
    }
    if (usedScopesAt(op, s0, reads) != usedScopesAt(op, s1, reads)) {
        return false;
    }

    // First insert temporarily Eval node to the AST, then perform dependence
    // analysis

    InsertTmpEval inserter(s0Expr, s1Expr, s0Side, s0, s1Side, s1);
    auto tmpOp = inserter(op);
    tmpOp = flattenStmtSeq(tmpOp);
    ASSERT(inserter.s0Eval().isValid());
    ASSERT(inserter.s1Eval().isValid());

    auto s0Eval = findStmt(tmpOp, inserter.s0Eval());
    auto s1Eval = findStmt(tmpOp, inserter.s1Eval());
    if (s0Eval->nextStmt() == s1Eval) {
        return true; // early exit: the period to check is empty
    }

    auto common = lcaStmt(s0Eval, s1Eval);
    FindDepsDir dir;
    for (auto p = common; p.isValid(); p = p->parentStmt()) {
        if (p->nodeType() == ASTNodeType::For) {
            dir.emplace_back(p->id(), DepDirection::Same);
        }
    }

    auto outMostReachableStmtInsideVarDef = [](const Stmt &_s,
                                               const Stmt &vardef) {
        Stmt ret = _s;
        for (auto s = _s; s.isValid(); s = s->parentStmt()) {
            if (s->nodeType() == ASTNodeType::For) {
                ret = s;
            }
            if (s == vardef) {
                break;
            }
        }
        return ret;
    };

    // write -> serialized PBSet
    std::unordered_map<Stmt, std::string> writesWAR;
    std::mutex m;
    auto foundWAR = [&](const Dependence &dep) {
        // Serialize WAR map because it is from a random PBCtx
        auto strWAR =
            toString(apply(domain(dep.later2EarlierIter_), dep.laterIter2Idx_));
        // only lock for writing the map
        std::lock_guard l(m);
        writesWAR[dep.later_.stmt_] = strWAR;
    };
    FindDeps()
        .direction({dir})
        .type(DEP_WAR)
        .filterEarlier([&](const AccessPoint &earlier) {
            return earlier.stmt_->id() == inserter.s0Eval();
        })
        .filterLater([&](const AccessPoint &later) {
            return outMostReachableStmtInsideVarDef(common, later.def_)
                ->isAncestorOf(later.stmt_);
        })
        .noProjectOutPrivateAxis(true)(tmpOp, unsyncFunc(foundWAR));

    auto foundRAW = [&](const Dependence &dep) {
        // re-construct WAR map from stored string in current PBCtx
        auto w0 = PBSet(dep.presburger_, writesWAR[dep.earlier_.stmt_]);
        auto w1 = apply(range(dep.later2EarlierIter_), dep.earlierIter2Idx_);
        if (!intersect(std::move(w0), std::move(w1)).empty())
            throw ModifiedException{};
    };
    try {
        FindDeps()
            .direction({dir})
            .type(DEP_RAW)
            .filterLater([&](const AccessPoint &later) {
                return later.stmt_->id() == inserter.s1Eval();
            })
            .filterEarlier([&](const AccessPoint &earlier) {
                return writesWAR.contains(earlier.stmt_);
            })
            .noProjectOutPrivateAxis(true)(tmpOp, unsyncFunc(foundRAW));
    } catch (const ModifiedException &) {
        return false;
    }
    return true;
}

bool checkNotModified(const Stmt &op, const Expr &expr,
                      CheckNotModifiedSide s0Side, const ID &s0,
                      CheckNotModifiedSide s1Side, const ID &s1) {
    return checkNotModified(op, expr, expr, s0Side, s0, s1Side, s1);
}

} // namespace freetensor
