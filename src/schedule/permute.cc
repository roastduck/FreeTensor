#include <functional>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <math/parse_pb_expr.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
#include <pass/shrink_for.h>
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
            Expr condition = reversePermute_.cond_;
            if (!condition.isValid())
                condition = makeBoolConst(true);
            for (auto &&[i, l] : iter::enumerate(loopsId_)) {
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
            inner = (*this)(inner);
            inner = makeIf(ID{}, condition, inner);
            for (auto &&newIter : iter::reversed(reversePermute_.args_)) {
                inner = makeFor(ID{}, newIter, makeIntConst(INT32_MIN),
                                makeIntConst(INT32_MAX), makeIntConst(1),
                                makeIntConst(int64_t(INT32_MAX) - INT32_MIN),
                                Ref<ForProperty>::make(), inner);
                permutedLoopsId_.push_front(inner->id());
            }
            iterReplacer_.clear();
            return inner;
        } else {
            return Mutator::visit(op);
        }
    }

    Expr visit(const Var &op) override {
        if (iterReplacer_.find(op->name_) != iterReplacer_.end())
            return iterReplacer_[op->name_];
        return op;
    }
};

std::pair<Stmt, std::vector<ID>> permute(
    const Stmt &_ast, const std::vector<ID> &loopsId,
    const std::function<std::vector<Expr>(std::vector<Expr>)> &transformFunc) {

    if (loopsId.size() == 0)
        throw InvalidSchedule("No loop is specified");

    // flatten the AST first since we expect perfectly nested loops
    auto ast = flattenStmtSeq(_ast);

    // look for the loops specified
    CheckLoopOrder checker(loopsId);
    checker(ast);
    const auto &loops = checker.order();

    // some sanity check
    {
        auto error = [&](const std::string &content) {
            std::string msg = "Loops ";
            for (auto &&[i, item] : iter::enumerate(loopsId)) {
                msg += (i > 0 ? ", " : "") + toString(item);
            }
            msg += " ";
            msg += content;
            throw InvalidSchedule(msg);
        };

        if (checker.stmtSeqInBetween().size() > 0)
            error(
                "should be perfectly nested without any statements in between");

        for (auto &&[i, item] : iter::enumerate(loops))
            if (loopsId[i] != item->id())
                error("should be in the same order as the input");
    }

    // prepare the permutation map
    std::string permuteMapStr;
    {
        auto originalIter =
            iter::range(loops.size()) | iter::imap([](auto &&i) {
                return makeVar("din" + std::to_string(i))
                    .template as<VarNode>();
            });
        auto permutedIter =
            transformFunc({originalIter.begin(), originalIter.end()});

        // check for loads in the permuted iteration expressions
        for (auto &&[i, expr] : iter::enumerate(permutedIter))
            if (allReads(expr).size() > 0)
                throw InvalidSchedule(
                    "Unexpected load in component " + std::to_string(i) +
                    " of provided transform, which is " + toString(expr));

        // start serializing the map
        std::ostringstream oss;

        oss << "{[";
        // sources
        for (auto &&[i, item] : iter::enumerate(originalIter)) {
            if (i > 0)
                oss << ", ";
            oss << mangle(item->name_);
        }

        oss << "] -> [";

        // destinations
        // give them names here since they are not in the permutedIter
        for (auto i : iter::range(permutedIter.size())) {
            if (i > 0)
                oss << ", ";
            oss << "dout" << i;
        }

        oss << "]: ";

        // generate presburger expressions for original iter -> permuted iter
        GenPBExpr genPBExpr;
        for (auto &&[i, item] : iter::enumerate(permutedIter)) {
            if (i > 0)
                oss << " and ";

            auto res = genPBExpr.gen(item);
            if (!res.isValid())
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

    auto dir = loopsId | iter::imap([&](auto &&l) -> FindDepsDir {
                   auto d = evenOuterLoops;
                   d.emplace_back(l, DepDirection::Different);
                   return d;
               });

    // map for extracting specified dimensions
    auto getExtractDimMap = [](int begin, int end, int total) {
        std::ostringstream oss;
        oss << "{[";
        for (auto i : iter::range(total))
            oss << (i > 0 ? ", " : "") << "din" << i;
        oss << "] -> [";
        for (auto i : iter::range(begin, end))
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
        .noProjectOutProvateAxis(true)
        .scope2CoordCallback(
            [&](const std::unordered_map<ID, std::vector<IterAxis>>
                    &scope2coord) {
                numBackDims = scope2coord.at(loopsId[0]).size() - 1;
                auto innerMostAxes = scope2coord.at(loopsId.back());
                ASSERT(numBackDims + loops.size() == innerMostAxes.size());

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
                PBMap real2iter(pbCtx, "{[" + ossRaw.str() + "] -> [" +
                                           ossIter.str() + "]}");
                iter2permuted =
                    toString(applyRange(reverse(real2iter), permuteMap));
            })
        .filterSubAST(loops.front()->id())(ast, [&](const Dependency &d) {
            auto iter2permutedMap = PBMap(d.presburger_, iter2permuted);
            auto &&permuteMapLater = applyRange(
                PBMap(d.presburger_,
                      getExtractDimMap(numBackDims, numBackDims + loops.size(),
                                       d.later_.iter_.size())),
                iter2permutedMap);
            auto &&permuteMapEarlier = applyRange(
                PBMap(d.presburger_,
                      getExtractDimMap(numBackDims, numBackDims + loops.size(),
                                       d.earlier_.iter_.size())),
                iter2permutedMap);
            auto permutedLater2Earlier =
                applyRange(applyDomain(d.later2EarlierIter_, permuteMapLater),
                           permuteMapEarlier);
            PBSpace space(range(permutedLater2Earlier));
            ASSERT(space == PBSpace(domain(permutedLater2Earlier)));
            auto violated = intersect(permutedLater2Earlier, lexLE(space));
            if (!violated.empty())
                throw InvalidSchedule(
                    "Provided transformation violates dependency");
            //! TODO: more diagnostics
        });

    auto reversePermute =
        parseSimplePBFunc(toString(PBFunc(reverse(permuteMap))));

    Permute permuter(loopsId, reversePermute);
    ast = permuter(ast);
    ast = shrinkFor(ast);

    return {
        ast,
        {permuter.permutedLoopsId().begin(), permuter.permutedLoopsId().end()}};
}

} // namespace freetensor
