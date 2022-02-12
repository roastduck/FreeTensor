#ifndef VECTORIZE_H
#define VECTORIZE_H

#include <mutator.h>

namespace ir {

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

} // namespace ir

#endif // VECTORIZE_H
