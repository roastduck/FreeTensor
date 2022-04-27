#ifndef FREE_TENSOR_VECTORIZE_H
#define FREE_TENSOR_VECTORIZE_H

#include <mutator.h>

namespace freetensor {

class Vectorize : public Mutator {
    ID loop_;
    bool done_ = false;

  public:
    Vectorize(const ID &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visit(const For &op) override;
};

Stmt vectorize(const Stmt &ast, const ID &loop);

} // namespace freetensor

#endif // FREE_TENSOR_VECTORIZE_H
