#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>

namespace ir {

class Parallelize : public Mutator {
    std::string loop_, parallel_;
    std::vector<std::string> outerLoops_, loopStack_;
    bool done_ = false;

    // For GPU threads, we may want to bind a loop to threadIdx.x inside another
    // loop bound to threadIdx.x. For example, we may implement a matmul with
    // collaborative fetch as below:
    //
    // for i : threadIdx.x
    //   for j : threadIdx.y
    //     for k : threadIdx.y
    //       load block A[i, k]
    //     for k : thradIdx.x
    //       load block B[k, j]
    //     for k
    //       compute block C[i, j] = A[i, k] * B[k, j]
    //
    // We have no way to separate these nested loops. However, some nesting is
    // illegal. For example:
    //
    // for i : threadIdx.x
    //   for j : threadIdx.x
    //     A[i, j] ++
    //
    // This is illegal because it violates its serial semantics. The following
    // variables are used to check this behaviour
    std::unordered_map<std::string, std::string> para2var_;
    std::unordered_set<std::string> hiddenVars_;

  public:
    Parallelize(const std::string &loop, const std::string &parallel)
        : loop_(loop), parallel_(parallel) {}

    bool done() const { return done_; }
    const std::vector<std::string> outerLoops() const { return outerLoops_; }

  protected:
    Stmt visit(const For &op) override;
    Expr visit(const Var &op) override;
};

Stmt parallelize(const Stmt &ast, const std::string &loop,
                 const std::string &parallel);

} // namespace ir

#endif // PARALLELIZE_H
