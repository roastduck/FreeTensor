#include <analyze/find_stmt.h>
#include <pass/hoist_var_over_stmt_seq.h>
#include <schedule/fission.h>
#include <schedule/move_to.h>
#include <schedule/swap.h>

namespace freetensor {

std::pair<Stmt, std::pair<ID, ID>> moveTo(const Stmt &_ast, const ID &_stmt,
                                          MoveToSide side, const ID &_dst) {
    auto ast = _ast;
    auto stmt = _stmt, dst = _dst;
    auto stmtBody = stmt;
    while (true) {
        ast = hoistVarOverStmtSeq(ast);
        Stmt s = findStmt(ast, stmt);
        Stmt d = findStmt(ast, dst);

        auto movingUp = [&]() {
            if (d->isAncestorOf(s)) {
                return side == MoveToSide::Before;
            }
            if (auto prev = s->prevStmt(); prev.isValid()) {
                return d->isBefore(side == MoveToSide::After ? prev : s);
            } else {
                return d->isBefore(s);
            }
        };
        auto movingDown = [&]() {
            if (d->isAncestorOf(s)) {
                return side == MoveToSide::After;
            }
            if (auto next = s->nextStmt(); next.isValid()) {
                return (side == MoveToSide::Before ? next : s)->isBefore(d);
            } else {
                return s->isBefore(d);
            }
        };

        if (movingUp()) {
            if (s->prevStmt().isValid()) {
                std::vector<ID> orderRev;
                while (s->prevStmt().isValid() && movingUp()) {
                    s = s->prevStmt();
                    orderRev.emplace_back(s->id());
                }
                orderRev.emplace_back(stmt);
                std::vector<ID> order(orderRev.rbegin(), orderRev.rend());
                ast = swap(ast, order);
            } else {
                while (!s->prevStmt().isValid() && movingUp()) {
                    s = s->parentCtrlFlow();
                }
                if (s->nodeType() != ASTNodeType::For) {
                    throw InvalidSchedule(
                        ast, "Fission a If node in a StmtSeq is not currently "
                             "supported in moveTo");
                    // TODO: Fission IfNode
                }
                auto &&[newAst, idMap] =
                    fission(ast, s->id(), FissionSide::After, stmt, ".a", "");
                ast = newAst;
                auto &&[idMapBefore, idMapAfter] = idMap;
                stmtBody = idMapBefore.at(stmt);
                stmt = idMapBefore.at(s->id());
            }
            // TODO: Fuse if d is inner of s

        } else if (movingDown()) {
            if (s->nextStmt().isValid()) {
                std::vector<ID> order;
                while (s->nextStmt().isValid() && movingDown()) {
                    s = s->nextStmt();
                    order.emplace_back(s->id());
                }
                order.emplace_back(stmt);
                ast = swap(ast, order);
            } else {
                while (!s->nextStmt().isValid() && movingDown()) {
                    s = s->parentCtrlFlow();
                }
                if (s->nodeType() != ASTNodeType::For) {
                    throw InvalidSchedule(
                        ast, "Fission a If node in a StmtSeq is not currently "
                             "supported in moveTo");
                    // TODO: Fission IfNode
                }
                // Leave IDs of the other statements unchanged
                auto &&[newAst, idMap] =
                    fission(ast, s->id(), FissionSide::Before, stmt, "", ".b");
                ast = newAst;
                auto &&[idMapBefore, idMapAfter] = idMap;
                stmtBody = idMapAfter.at(stmt);
                stmt = idMapAfter.at(s->id());
            }
            // TODO: Fuse if d is inner of s

        } else {
            return {ast, {stmtBody, stmt}};
        }
    }
}

} // namespace freetensor
