#ifndef MAKE_ATOMIC_H
#define MAKE_ATOMIC_H

#include <unordered_set>
#include <vector>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllParallel : public Visitor {
    std::vector<std::string> results_;

  public:
    const std::vector<std::string> &results() const { return results_; }

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

inline Func makeAtomic(const Func &func) {
    return makeFunc(func->name_, func->params_, makeAtomic(func->body_),
                    func->src_);
}

} // namespace ir

#endif // MAKE_ATOMIC_H
