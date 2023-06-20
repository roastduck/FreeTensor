#include <analyze/deps.h>
#include <analyze/find_loop_variance.h>
#include <analyze/find_stmt.h>
#include <pass/merge_and_hoist_if.h>
#include <schedule.h>
#include <schedule/fission.h>
#include <schedule/hoist_selected_var.h>

namespace freetensor {

Stmt AddDimToVar::visit(const For &op) {
    forMap_[op->id()] = op;
    return BaseClass::visit(op);
}

Stmt AddDimToVar::visit(const VarDef &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::VarDef);

    auto op = __op.as<VarDefNode>();
    if (toAdd_.count(op->id())) {
        op->buffer_ = deepCopy(op->buffer_);
        auto &shape = op->buffer_->tensor()->shape();
        for (auto &&loop : views::reverse(toAdd_.at(op->id()))) {
            shape.insert(shape.begin(), forMap_.at(loop)->len_);
        }
    }
    return op;
}

Stmt AddDimToVar::visit(const Store &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Store);
    return doAdd(__op.as<StoreNode>());
}

Expr AddDimToVar::visit(const Load &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::Load);
    return doAdd(__op.as<LoadNode>());
}

Stmt AddDimToVar::visit(const ReduceTo &_op) {
    auto __op = BaseClass::visit(_op);
    ASSERT(__op->nodeType() == ASTNodeType::ReduceTo);
    return doAdd(__op.as<ReduceToNode>());
}

Stmt FissionFor::visitStmt(const Stmt &op) {
    if (!inside_) {
        return Mutator::visitStmt(op);
    } else {
        auto oldAnyInside = anyInside_;
        anyInside_ = false;
        isAfter_ |= op->id() == before_;
        anyInside_ |= (isPart0_ && !isAfter_) || (!isPart0_ && isAfter_);
        auto ret = Mutator::visitStmt(op);
        isAfter_ |= op->id() == after_;
        if (!anyInside_) {
            ret = makeStmtSeq({});
        }
        anyInside_ |= oldAnyInside;
        return ret;
    }
}

void FissionFor::markNewId(const Stmt &op, bool isPart0) {
    ID oldId = op->id();
    if (isPart0 ? op0_ : op1_)
        op->setId();
    ID newId = op->id();
    if (isPart0) {
        if (op0_)
            op->metadata() = makeMetadata(*op0_, op);
        ids0_.emplace(oldId, newId);
    } else {
        if (op1_)
            op->metadata() = makeMetadata(*op1_, op);
        ids1_.emplace(oldId, newId);
    }
}

Stmt FissionFor::visit(const For &op) {
    if (op->id() != loop_) {
        auto ret = Mutator::visit(op);
        if (inside_) {
            markNewId(ret, isPart0_);
        }
        return ret;
    } else {
        auto begin = (*this)(op->begin_);
        auto end = (*this)(op->end_);
        auto step = (*this)(op->step_);
        auto len = (*this)(op->len_);
        inside_ = true;
        isPart0_ = true, isAfter_ = false, anyInside_ = false;
        auto part0 = (*this)(op->body_);
        isPart0_ = false, isAfter_ = false, anyInside_ = false;
        auto part1 = (*this)(op->body_);
        inside_ = false;
        auto for0 = makeFor(op->iter_, begin, end, step, len, op->property_,
                            part0, op->metadata(), op->id());
        auto for1 = makeFor(op->iter_, begin, end, step, len, op->property_,
                            part1, op->metadata(), op->id());
        markNewId(for0, true);
        markNewId(for1, false);
        return makeStmtSeq({for0, for1});
    }
}

Stmt FissionFor::visit(const StmtSeq &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const VarDef &_op) {
    if (!inside_) {
        return Mutator::visit(_op);
    } else {
        varUses_.erase(_op->name_);
        auto __op = Mutator::visit(_op);
        ASSERT(__op->nodeType() == ASTNodeType::VarDef);
        auto op = __op.as<VarDefNode>();
        Stmt ret = varUses_.count(op->name_) ? __op : (Stmt)op->body_;
        markNewId(ret, isPart0_);
        return ret;
    }
}

Stmt FissionFor::visit(const Store &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Expr FissionFor::visit(const Load &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    return Mutator::visit(op);
}

Stmt FissionFor::visit(const ReduceTo &op) {
    if (inPart()) {
        varUses_.insert(op->var_);
    }
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const If &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

Stmt FissionFor::visit(const Assert &op) {
    auto ret = Mutator::visit(op);
    if (inside_) {
        markNewId(ret, isPart0_);
    }
    return ret;
}

std::pair<Stmt,
          std::pair<std::unordered_map<ID, ID>, std::unordered_map<ID, ID>>>
fission(const Stmt &_ast, const ID &loop, FissionSide side, const ID &splitter,
        bool allowEnlarge, const std::string &suffix0,
        const std::string &suffix1) {
    if (suffix0.empty() && suffix1.empty())
        throw InvalidSchedule(
            "Cannot preserve ID and Metadata for both first and second parts");

    Stmt splitterNode;
    try {
        splitterNode = findStmt(_ast, splitter);
    } catch (const UnexpectedQueryResult &e) {
        throw InvalidSchedule("Split point " + toString(splitter) +
                              " not found inside " + toString(loop));
    }
    ID leftOfSplitter, rightOfSplitter;
    if (side == FissionSide::Before) {
        rightOfSplitter = splitter;
        if (auto node = splitterNode->prevStmtInDFSOrder();
            node.isValid() && node->ancestorById(loop).isValid()) {
            leftOfSplitter = node->id();
        }
    } else {
        leftOfSplitter = splitter;
        if (auto node = splitterNode->nextStmtInDFSOrder();
            node.isValid() && node->ancestorById(loop).isValid()) {
            rightOfSplitter = node->id();
        }
    }

    Stmt ast = _ast;
    if (leftOfSplitter.isValid() && rightOfSplitter.isValid()) {
        // Non-trivial fission

        // Select everything in `loop` but crossing `splitter`
        std::string selectCrossing = "<<-" + toString(loop) + "&->>" +
                                     toString(leftOfSplitter) + "&->>" +
                                     toString(rightOfSplitter);
        std::vector<ID> affectedLoops = {loop}; // From outer to inner
        for (auto &&subloop : findAllStmt(ast, "<For>&" + selectCrossing)) {
            affectedLoops.emplace_back(subloop->id());
        }
        ast = hoistSelectedVar(ast, selectCrossing);

        auto variants = findLoopVariance(ast);

        std::vector<FindDepsDir> disjunct;
        for (auto &&inner : affectedLoops) {
            FindDepsDir dir = {{inner, DepDirection::Normal}};
            for (auto outer = findStmt(ast, loop)->parentStmt();
                 outer.isValid(); outer = outer->parentStmt()) {
                if (outer->nodeType() == ASTNodeType::For) {
                    dir.emplace_back(outer->id(), DepDirection::Same);
                }
            }
            disjunct.emplace_back(std::move(dir));
        }
        auto isRealWrite = [&](const ID &loop, const VarDef &def) -> bool {
            return isVariant(variants.second, def, loop);
        };
        std::unordered_map<ID, std::vector<ID>> toAdd;
        auto found = [&](const Dependence &d) {
            auto &&id = d.dir_[0].first.id_;
            if (!findStmt(_ast, d.defId())->ancestorById(id).isValid()) {
                // The variable is NOT a local variable inside the loop being
                // fissioned, which means the value of the variable may be
                // passed in to the first iteration, or passed out from the last
                // iteration. Solving this type of conflict by adding dimensions
                // is non-trivial. (TODO: We can support this in the future in
                // the way like `firstprivate` and `lastprivate` in OpenMP)
                throw InvalidSchedule(toString(d) + " cannot be resolved");
            }
            if (allowEnlarge) {
                if (!isRealWrite(id, d.def()) &&
                    d.earlier()->nodeType() == ASTNodeType::Load) {
                    return;
                }
                if (!isRealWrite(id, d.def()) &&
                    d.later()->nodeType() == ASTNodeType::Load) {
                    return;
                }
                if (std::find(toAdd[d.defId()].begin(), toAdd[d.defId()].end(),
                              id) == toAdd[d.defId()].end()) {
                    toAdd[d.defId()].emplace_back(id);
                }
            } else {
                throw InvalidSchedule(toString(d) + " cannot be resolved");
            }
        };
        auto leftOfSplitterStmt = findStmt(ast, leftOfSplitter);
        auto rightOfSplitterStmt = findStmt(ast, rightOfSplitter);
        FindDeps()
            .direction(disjunct)
            .filterSubAST(loop)
            .filterEarlier([&](const AccessPoint &earlier) {
                // Reverse dependence: earlier at the second fissioned part
                return earlier.stmt_->isAncestorOf(leftOfSplitterStmt) ||
                       leftOfSplitterStmt->isBefore(earlier.stmt_);
            })
            .filterLater([&](const AccessPoint &later) {
                // Reverse dependence: later at the first fissioned part
                return later.stmt_->isAncestorOf(rightOfSplitterStmt) ||
                       later.stmt_->isBefore(rightOfSplitterStmt);
            })(ast, found);

        AddDimToVar adder(toAdd);
        ast = adder(ast);
    }

    auto before = side == FissionSide::Before ? splitter : ID{};
    auto after = side == FissionSide::After ? splitter : ID{};
    FissionFor mutator(loop, before, after, suffix0, suffix1);
    ast = mutator(ast);
    ast = mergeAndHoistIf(ast);

    auto ids0 = mutator.ids0();
    auto ids1 = mutator.ids1();
    if (findAllStmt(ast, ids0.at(loop)).empty()) {
        ids0.clear();
    }
    if (findAllStmt(ast, ids1.at(loop)).empty()) {
        ids1.clear();
    }

    return {ast, {ids0, ids1}};
}

std::pair<Schedule::IDMap, Schedule::IDMap>
Schedule::fission(const ID &loop, FissionSide side, const ID &splitter,
                  bool allowEnlarge, const std::string &suffix0,
                  const std::string &suffix1) {
    beginTransaction();
    auto log =
        appendLog(MAKE_SCHEDULE_LOG(Fission, freetensor::fission, loop, side,
                                    splitter, allowEnlarge, suffix0, suffix1));
    try {
        auto ret = applyLog(log);
        commitTransaction();
        return ret;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(log, ast(), e.what());
    }
}

} // namespace freetensor
