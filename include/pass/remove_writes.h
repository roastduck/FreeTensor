#ifndef REMOVE_WRITES_H
#define REMOVE_WRITES_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllStmtSeq : public Visitor {
    std::vector<std::string> ids_;

  public:
    const std::vector<std::string> ids() const { return ids_; }

  protected:
    void visit(const StmtSeq &op) override {
        Visitor::visit(op);
        ids_.emplace_back(op->id());
    }
};

class RemoveWrites : public Mutator {
    const std::unordered_set<Stmt> &redundant_;
    const std::unordered_map<Stmt, Stmt> &replacement_;

  public:
    RemoveWrites(const std::unordered_set<Stmt> &redundant,
                 const std::unordered_map<Stmt, Stmt> &replacement)
        : redundant_(redundant), replacement_(replacement) {}

    template <class T> Stmt doVisit(const T &op) {
        if (redundant_.count(op)) {
            return makeStmtSeq(op->id(), {});
        } else if (replacement_.count(op)) {
            return replacement_.at(op);
        } else {
            return Mutator::visit(op);
        }
    }

  protected:
    Stmt visit(const Store &op) override { return doVisit(op); }
    Stmt visit(const ReduceTo &op) override { return doVisit(op); }
};

/**
 * Remove redundant writes
 *
 * E.g.
 *
 * ```
 * x[0] = 1;
 * x[1] = 2;
 * ```
 */
Stmt removeWrites(const Stmt &op);

} // namespace ir

#endif // REMOVE_WRITES_H
