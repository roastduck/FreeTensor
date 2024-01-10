#include <schedule.h>

namespace freetensor {

std::pair<ID, ID> Schedule::moveTo(const ID &_stmt, MoveToSide side,
                                   const ID &_dst) {
    beginTransaction();
    try {
        auto stmt = _stmt, dst = _dst;
        auto stmtBody = stmt;
        while (true) {
            Stmt s = findStmt(ast(), stmt);
            Stmt d = findStmt(ast(), dst);

            auto movingUp = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::Before;
                }
                if (auto prev = s->prevInCtrlFlow(); prev.isValid()) {
                    return d->isBefore(side == MoveToSide::After ? prev : s);
                } else {
                    return d->isBefore(s);
                }
            };
            auto movingDown = [&]() {
                if (d->isAncestorOf(s)) {
                    return side == MoveToSide::After;
                }
                if (auto next = s->nextInCtrlFlow(); next.isValid()) {
                    return (side == MoveToSide::Before ? next : s)->isBefore(d);
                } else {
                    return s->isBefore(d);
                }
            };

            if (movingUp()) {
                if (s->prevInCtrlFlow().isValid()) {
                    std::vector<ID> orderRev;
                    while (s->prevInCtrlFlow().isValid() && movingUp()) {
                        s = s->prevInCtrlFlow();
                        orderRev.emplace_back(s->id());
                    }
                    orderRev.emplace_back(stmt);
                    std::vector<ID> order(orderRev.rbegin(), orderRev.rend());
                    swap(order);
                } else {
                    while (!s->prevInCtrlFlow().isValid() && movingUp()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast(), FT_MSG << "Fission a " << s->nodeType()
                                          << " node in a StmtSeq is not "
                                             "currently supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    auto idMapBefore = fission(s->id(), FissionSide::After,
                                               stmt, true, ".a", "")
                                           .first;
                    stmtBody = idMapBefore.at(stmt);
                    stmt = idMapBefore.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else if (movingDown()) {
                if (s->nextInCtrlFlow().isValid()) {
                    std::vector<ID> order;
                    while (s->nextInCtrlFlow().isValid() && movingDown()) {
                        s = s->nextInCtrlFlow();
                        order.emplace_back(s->id());
                    }
                    order.emplace_back(stmt);
                    swap(order);
                } else {
                    while (!s->nextInCtrlFlow().isValid() && movingDown()) {
                        s = s->parentCtrlFlow();
                    }
                    if (s->nodeType() != ASTNodeType::For) {
                        throw InvalidSchedule(
                            ast(), FT_MSG << "Fission a " << s->nodeType()
                                          << " node in a StmtSeq is not "
                                             "currently supported in moveTo");
                        // TODO: Fission IfNode
                    }
                    // Leave IDs of the other statements unchanged
                    auto idMapAfter = fission(s->id(), FissionSide::Before,
                                              stmt, true, "", ".b")
                                          .second;
                    stmtBody = idMapAfter.at(stmt);
                    stmt = idMapAfter.at(s->id());
                }
                // TODO: Fuse if d is inner of s

            } else {
                commitTransaction();
                return {stmtBody, stmt};
            }
        }
    } catch (const InvalidSchedule &e) {
        abortTransaction();
        throw InvalidSchedule(ast(), FT_MSG << "Invalid move_to(" << _stmt
                                            << ", " << _dst
                                            << "): " << e.what());
    }
}

} // namespace freetensor
