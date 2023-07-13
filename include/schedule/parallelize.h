#ifndef FREE_TENSOR_PARALLELIZE_H
#define FREE_TENSOR_PARALLELIZE_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>

namespace freetensor {

class Parallelize : public Mutator {
    ID loop_;
    ParallelScope parallel_;
    std::vector<ID> outerLoops_, loopStack_;
    bool done_ = false;

  public:
    Parallelize(const ID &loop, const ParallelScope &parallel)
        : loop_(loop), parallel_(parallel) {}

    bool done() const { return done_; }
    const std::vector<ID> outerLoops() const { return outerLoops_; }

  protected:
    Stmt visit(const For &op) override;
};

Stmt parallelize(const Stmt &ast, const ID &loop, const ParallelScope &parallel,
                 bool allowReduction);

} // namespace freetensor

#endif // FREE_TENSOR_PARALLELIZE_H
