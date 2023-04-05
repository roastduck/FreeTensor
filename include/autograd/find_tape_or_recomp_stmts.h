#ifndef FREE_TENSOR_FIND_TAPE_OR_RECOMP_STMTS_H
#define FREE_TENSOR_FIND_TAPE_OR_RECOMP_STMTS_H

#include <unordered_set>

#include <autograd/derivative.h>

namespace freetensor {

/**
 * Given a set of variables that we are willing to tape, find out all the
 * statements we need to tape, and all the statements we need to recompute
 *
 * @return : ({ID of VarDef -> IDs of statements to tape}, {ID of VarDef -> IDs
 * of statements to recompute})
 */
std::pair<std::unordered_map<ID, std::unordered_set<ID>>,
          std::unordered_map<ID, std::unordered_set<ID>>>
findTapeOrRecompStmts(
    const Stmt &op, const std::unordered_set<ID> &defsToTape,
    const std::unordered_set<ID> defsNeedGrad,
    std::unordered_map<StmtOrExprID, Derivative::LazyFullDerivative>
        &derivatives);

} // namespace freetensor

#endif // FREE_TENSOR_FIND_TAPE_OR_RECOMP_STMTS_H
