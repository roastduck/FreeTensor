#include <functional>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <math/parse_pb_expr.h>
#include <pass/shrink_for.h>
#include <schedule.h>
#include <schedule/check_loop_order.h>
#include <schedule/permute.h>
#include <serialize/mangle.h>

namespace freetensor {

class Permute : public Mutator {
    const std::vector<ID> &loopsId_;
    const SimplePBFuncAST &reversePermute_;
    std::unordered_map<std::string, Expr> iterReplacer_;
    std::deque<ID> permutedLoopsId_;

  public:
    Permute(const std::vector<ID> &loopsId,
            const SimplePBFuncAST &reversePermute)
        : loopsId_(loopsId), reversePermute_(reversePermute), iterReplacer_() {}

    const std::deque<ID> &permutedLoopsId() const { return permutedLoopsId_; }

  protected:
    Stmt visit(const For &op) override {
        if (op->id() == loopsId_.front()) {
            Stmt inner = op;
            // the condition from PBFunc; if nothing, use true
            Expr condition = reversePermute_.cond_;
            if (!condition.isValid())
                condition = makeBoolConst(true);
            // iterate over the loops
            for (auto &&[i, l] : views::enumerate(loopsId_)) {
                // sanity check
                ASSERT(inner->id() == l &&
                       inner->nodeType() == ASTNodeType::For);
                auto innerFor = inner.as<ForNode>();
                // construct condition
                condition = makeLAnd(
                    condition,
                    makeIfExpr(makeGT(innerFor->step_, makeIntConst(0)),
                               makeLAnd(makeLE(innerFor->begin_,
                                               reversePermute_.values_[i]),
                                        makeLT(reversePermute_.values_[i],
                                               innerFor->end_)),
                               makeLAnd(makeGE(innerFor->begin_,
                                               reversePermute_.values_[i]),
                                        makeGT(reversePermute_.values_[i],
                                               innerFor->end_))));
                // record expression to replace the iter Var
                iterReplacer_[innerFor->iter_] = reversePermute_.values_[i];
                inner = innerFor->body_;
            }
            // inner is now innermost->body_; transform `Var's in it
            inner = (*this)(inner);
            // wrap with the conditions
            inner = makeIf(condition, inner);
            // wrap with the new loops
            for (auto &&newIter : views::reverse(reversePermute_.args_)) {
                inner = makeFor(newIter, makeIntConst(INT32_MIN),
                                makeIntConst(INT32_MAX), makeIntConst(1),
                                makeIntConst(int64_t(INT32_MAX) - INT32_MIN),
                                Ref<ForProperty>::make(), inner);
                permutedLoopsId_.push_front(inner->id());
            }
            // clear replacing map, later `Var's won't be transformed
            iterReplacer_.clear();
            return inner;
        } else {
            return Mutator::visit(op);
        }
    }

    Expr visit(const Var &op) override {
        // lookup the map; if exists, replace iter with provided expression
        if (auto it = iterReplacer_.find(op->name_); it != iterReplacer_.end())
            return it->second;
        return op;
    }
};

std::pair<Stmt, std::vector<ID>> permute(
    const Stmt &_ast, const std::vector<ID> &loopsId,
    const std::function<std::vector<Expr>(std::vector<Expr>)> &transformFunc) {

    if (loopsId.size() == 0)
        throw InvalidSchedule("No loop is specified");

    auto ast = _ast;

    // look for the loops specified
    CheckLoopOrder checker(loopsId);
    checker(ast);
    const auto &loops = checker.order();

    // some sanity check
    {
        auto error = [&](const std::string &content) {
            std::string msg = "Loops ";
            for (auto &&[i, item] : views::enumerate(loopsId)) {
                msg += (i > 0 ? ", " : "") + toString(item);
            }
            msg += " ";
            msg += content;
            throw InvalidSchedule(msg);
        };

        if (checker.stmtSeqInBetween().size() > 0)
            error(
                "should be perfectly nested without any statements in between");

        for (auto &&[i, item] : views::enumerate(loops))
            if (loopsId[i] != item->id())
                error("should be in the same order as the input");
    }

    // prepare the permutation map
    std::string permuteMapStr;
    {
        auto originalIter = views::ints(0ul, loops.size()) |
                            views::transform([](auto &&i) {
                                return makeVar("din" + std::to_string(i));
                            }) |
                            ranges::to<std::vector>();
        auto permutedIter = transformFunc(originalIter);

        // check for loads in the permuted iteration expressions
        for (auto &&[i, expr] : views::enumerate(permutedIter))
            if (allReads(expr).size() > 0)
                throw InvalidSchedule(
                    "Unexpected load in component " + std::to_string(i) +
                    " of provided transform, which is " + toString(expr));

        // start serializing the map
        std::ostringstream oss;

        oss << "{[";
        // sources
        for (auto &&[i, item] : views::enumerate(originalIter)) {
            if (i > 0)
                oss << ", ";
            oss << mangle(item.as<VarNode>()->name_);
        }

        oss << "] -> [";

        // destinations
        // give them names here since they are not in the permutedIter
        for (auto i : views::ints(0ul, permutedIter.size())) {
            if (i > 0)
                oss << ", ";
            oss << "dout" << i;
        }

        oss << "]: ";

        // generate presburger expressions for original iter -> permuted iter
        GenPBExpr genPBExpr;
        for (auto &&[i, item] : views::enumerate(permutedIter)) {
            if (i > 0)
                oss << " and ";

            auto res = genPBExpr.gen(item);
            if (!res.has_value())
                throw InvalidSchedule(
                    "Cannot generate Presburger expression for component " +
                    std::to_string(i) + " of provided transform, which is " +
                    toString(item));

            oss << "dout" << i << " = " << *res;
        }
        // finish serializing the map
        oss << "}";
        permuteMapStr = oss.str();
    }
    // check if the map is bijective
    PBCtx pbCtx;
    PBMap permuteMap(pbCtx, permuteMapStr);
    {
        //! FIXME: if step != 1 or -1, this check will be more strict than
        //! actually needed; need to add range restrictions to input dimensions
        if (!permuteMap.isBijective())
            throw InvalidSchedule("Provided transform is not bijective, which "
                                  "is required for permutation");
    }

    // Keep the even outer loops Same since we only operate in one iteration of
    // them
    FindDepsDir evenOuterLoops;
    for (auto node = loops[0]; node != nullptr;
         node = node->parentStmtByFilter([](const Stmt &stmt) {
                        return stmt->nodeType() == ASTNodeType::For;
                    })
                    .as<ForNode>())
        evenOuterLoops.emplace_back(node->id(), DepDirection::Same);

    auto dir = loopsId | views::transform([&](auto &&l) -> FindDepsDir {
                   auto d = evenOuterLoops;
                   d.emplace_back(l, DepDirection::Different);
                   return d;
               });

    // map for extracting specified dimensions
    auto getExtractDimMap = [](int begin, int end, int total) {
        std::ostringstream oss;
        oss << "{[";
        for (auto i : views::ints(0, total))
            oss << (i > 0 ? ", " : "") << "din" << i;
        oss << "] -> [";
        for (auto i : views::ints(begin, end))
            oss << (i > begin ? ", " : "") << "dout" << i - begin << " = din"
                << i;
        oss << "]}";
        return oss.str();
    };

    size_t numBackDims;
    std::string iter2permuted;

    FindDeps()
        .direction({dir.begin(), dir.end()})
        .filterAccess([&](const AccessPoint &ap) {
            return ap.def_->isAncestorOf(loops.front());
        })
        .noProjectOutPrivateAxis(true)
        .scope2CoordCallback(
            [&](const std::unordered_map<ID, std::vector<IterAxis>>
                    &scope2coord) {
                // compute number of backing dimensions, outer than our
                // outermost loop
                numBackDims = scope2coord.at(loopsId[0]).size() - 1;
                // sanity check: innermost should have exactly num. loops more
                // axes
                auto innerMostAxes = scope2coord.at(loopsId.back());
                ASSERT(numBackDims + loops.size() == innerMostAxes.size());

                // generate PB expr for realIter -> iter transformation
                GenPBExpr genPBExpr;
                std::ostringstream ossRaw, ossIter;
                for (size_t i = numBackDims; i < innerMostAxes.size(); i++) {
                    if (i > numBackDims) {
                        ossRaw << ", ";
                        ossIter << ", ";
                    }
                    ossRaw << *genPBExpr.gen(innerMostAxes[i].realIter_);
                    ossIter << *genPBExpr.gen(innerMostAxes[i].iter_);
                }
                // use anonymous output variables in ISL
                PBMap real2iter(pbCtx, "{[" + ossRaw.str() + "] -> [" +
                                           ossIter.str() + "]}");
                // prepare iter -> permuted map string, for later use in `found'
                iter2permuted =
                    toString(applyRange(reverse(real2iter), permuteMap));
            })
        .filterSubAST(loops.front()->id())(ast, [&](const Dependency &d) {
            // Construct map for iter -> permuted
            auto iter2permutedMap = PBMap(d.presburger_, iter2permuted);
            // laterIter -> permuted
            auto &&permuteMapLater = applyRange(
                PBMap(d.presburger_,
                      getExtractDimMap(numBackDims, numBackDims + loops.size(),
                                       d.later_.iter_.size())),
                iter2permutedMap);
            // earlierIter -> permuted
            auto &&permuteMapEarlier = applyRange(
                PBMap(d.presburger_,
                      getExtractDimMap(numBackDims, numBackDims + loops.size(),
                                       d.earlier_.iter_.size())),
                iter2permutedMap);
            // permuted(later) -> permuted(earlier)
            auto permutedLater2Earlier =
                applyRange(applyDomain(d.later2EarlierIter_, permuteMapLater),
                           permuteMapEarlier);
            // sanity check: two permuted should be in same space
            PBSpace space(range(permutedLater2Earlier));
            ASSERT(space == PBSpace(domain(permutedLater2Earlier)));
            // should never have permuted(later) <= permuted(earlier)
            auto violated = intersect(permutedLater2Earlier, lexLE(space));
            if (!violated.empty())
                throw InvalidSchedule(
                    "Provided transformation violates dependency");
            //! TODO: more diagnostics
        });

    // compute the function for reverse permute; we iterate against permuted
    // space, thus we need to compute the original iterators from the new ones
    auto reversePermute =
        parseSimplePBFunc(toString(PBFunc(reverse(permuteMap))));

    // perform transformation
    Permute permuter(loopsId, reversePermute);
    ast = permuter(ast);
    // shrinkFor since we use INT_MIN:INT_MAX range in Permute
    ast = shrinkFor(ast);

    return {
        ast,
        {permuter.permutedLoopsId().begin(), permuter.permutedLoopsId().end()}};
}

std::vector<ID> Schedule::permute(
    const std::vector<ID> &loopsId,
    const std::function<std::vector<Expr>(std::vector<Expr>)> &transformFunc) {
    beginTransaction();
    //! FIXME: put this into schedule logs
    try {
        auto ret = freetensor::permute(ast(), loopsId, transformFunc);
        setAst(quickOptimizations(ret.first));
        commitTransaction();
        return ret.second;
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw;
    }
}

} // namespace freetensor
