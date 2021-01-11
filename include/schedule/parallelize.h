#ifndef PARALLELIZE_H
#define PARALLELIZE_H

#include <mutator.h>

namespace ir {

class Parallelize : public Mutator {
    std::string loop_, parallel_;
    bool done_ = false;

  public:
    Parallelize(const std::string &loop, const std::string &parallel)
        : loop_(loop), parallel_(parallel) {}

    bool done() const { return done_; }

  protected:
    Stmt visit(const For &op) override;
};

} // namespace ir

#endif // PARALLELIZE_H

