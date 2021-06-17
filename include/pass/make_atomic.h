#ifndef MAKE_ATOMIC_H
#define MAKE_ATOMIC_H

#include <unordered_map>
#include <unordered_set>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllParallel : public Visitor {
    // Loop ID -> parallel type
    std::unordered_map<std::string, std::string> results_;

  public:
    const std::unordered_map<std::string, std::string> &results() const {
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

inline Func makeAtomic(const Func &func) {
    return makeFunc(func->name_, func->params_, makeAtomic(func->body_),
                    func->src_);
}

} // namespace ir

#endif // MAKE_ATOMIC_H
