#ifndef MAKE_ATOMIC_H
#define MAKE_ATOMIC_H

#include <unordered_map>
#include <unordered_set>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

struct ParallelInfo {
    std::string type_;                    // parallel type
    std::vector<std::string> outerLoops_; // outer loop ID
};

class FindAllParallel : public Visitor {
    // Loop ID -> ParallelInfo
    std::unordered_map<std::string, ParallelInfo> results_;

    std::vector<std::string> loopStack_;

  public:
    const std::unordered_map<std::string, ParallelInfo> &results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
};

class MakeAtomic : public Mutator {
    const std::unordered_set<std::string> &toAlter_;

  public:
    MakeAtomic(const std::unordered_set<std::string> &toAlter)
        : toAlter_(toAlter) {}

  protected:
    Stmt visit(const ReduceTo &op) override;
};

/**
 * Mark all racing ReduceTo nodes to be atomic
 */
Stmt makeAtomic(const Stmt &op);

DEFINE_PASS_FOR_FUNC(makeAtomic)

} // namespace ir

#endif // MAKE_ATOMIC_H
