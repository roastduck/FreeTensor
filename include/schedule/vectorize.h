#ifndef VECTORIZE_H
#define VECTORIZE_H

#include <mutator.h>

namespace ir {

class Vectorize : public Mutator {
    std::string loop_;
    bool done_ = false;

  public:
    Vectorize(const std::string &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    Stmt visit(const For &op) override;
};

} // namespace ir

#endif // VECTORIZE_H
