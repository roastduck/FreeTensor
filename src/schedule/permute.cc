#include <functional>

#include <analyze/all_uses.h>
#include <analyze/deps.h>
#include <pass/flatten_stmt_seq.h>
#include <pass/make_reduction.h>
#include <schedule/check_loop_order.h>
#include <schedule/permute.h>
#include <serialize/mangle.h>

namespace freetensor {

Stmt permute(
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
    std::string permuteMapPieces[3];
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

        // sources
        for (auto &&[i, item] : iter::enumerate(originalIter)) {
            if (i > 0)
                oss << ", ";
            oss << mangle(item->name_);
        }
        permuteMapPieces[0] = oss.str();
        oss.str("");

        // destinations
        // give them names here since they are not in the permutedIter
        for (auto i : iter::range(permutedIter.size())) {
            if (i > 0)
                oss << ", ";
            oss << "dout" << i;
        }
        permuteMapPieces[1] = oss.str();
        oss.str("");

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
        permuteMapPieces[2] = oss.str();
        oss.str("");
    }
    auto getPermuteMap = [&](int nOuterExtras, int nInnerExtras) {
        std::ostringstream oss;
        oss << "{[";
        for (auto &&i : iter::range(nOuterExtras))
            oss << "outerExtra" << i << ", ";
        oss << permuteMapPieces[0];
        for (auto &&i : iter::range(nInnerExtras))
            oss << ", innerExtra" << i;
        oss << "] -> [";
        for (auto &&i : iter::range(nOuterExtras))
            oss << "outerExtra" << i << ", ";
        oss << permuteMapPieces[1];
        // we project out inner extra dimensions here
        oss << "]: " << permuteMapPieces[2] << "}";
        return oss.str();
    };
    // check if the map is bijective
    {
        PBCtx pbCtx;
        PBMap permuteMap(pbCtx, getPermuteMap(0, 0));
        std::cout << "Permute map: " << permuteMap << std::endl;
        //! FIXME: if step != 1 or -1, this check will be more strict than
        //! actually needed; need to add range restrictions to input dimensions
        if (!permuteMap.isBijective())
            throw InvalidSchedule("Provided transform is not bijective, which "
                                  "is required for permutation");
    }

    auto getNumBackDims = [&](const AccessPoint &ap) {
        for (auto &&[i, iter] : iter::enumerate(ap.iter_)) {
            if (iter.iter_->nodeType() == ASTNodeType::Var &&
                iter.iter_.as<VarNode>()->name_ == loops[0]->iter_)
                return i;
        }
        ERROR("Outermost loop not found in iterator of found access point");
    };

    auto dir = loopsId | iter::imap([](auto &&l) -> FindDepsDir {
                   return {{l, DepDirection::Different}};
               });

    FindDeps()
        .direction({dir.begin(), dir.end()})
        // .filterAccess([&](const AccessPoint &ap) {
        //     return !ap.def_->ancestorById(loopsId.back()).isValid();
        // })
        .noProjectOutProvateAxis(true)
        .filterSubAST(loops.front()->id())(ast, [&](const Dependency &d) {
            std::cout << "Found dependency: " << d.later() << " over "
                      << d.earlier() << std::endl;
            auto numBackDims = getNumBackDims(d.earlier_);
            ASSERT(numBackDims == getNumBackDims(d.later_));
            std::cout << "later2earlier: " << d.later2EarlierIter_ << std::endl;
            auto permuteMapLater = PBMap(
                d.presburger_,
                getPermuteMap(numBackDims, d.later_.iter_.size() - numBackDims -
                                               loops.size()));
            auto permuteMapEarlier = PBMap(
                d.presburger_,
                getPermuteMap(numBackDims, d.earlier_.iter_.size() -
                                               numBackDims - loops.size()));
            auto permutedLater2Earlier =
                applyRange(applyDomain(d.later2EarlierIter_, permuteMapLater),
                           permuteMapEarlier);
            std::cout << "Permuted later2earlier: " << permutedLater2Earlier
                      << std::endl;
            PBSpace space(range(permutedLater2Earlier));
            ASSERT(space == PBSpace(domain(permutedLater2Earlier)));
            auto violated = intersect(permutedLater2Earlier, lexLE(space));
            if (!violated.empty()) {
                std::cout << "Violated map: " << violated << std::endl;
                throw InvalidSchedule(
                    "Provided transformation violates dependency");
            }
            //! TODO: more diagnostics
        });

    //! TODO: do the actual transformation

    return ast;
}

} // namespace freetensor
