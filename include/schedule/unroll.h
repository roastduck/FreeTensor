#ifndef UNROLL_H
#define UNROLL_H

#include <mutator.h>
#include <schedule.h>

namespace ir {

class Unroll : public Mutator {
    friend class Schedule;

    std::string loop_;
    bool done_ = false;

  public:
    Unroll(const std::string &loop) : loop_(loop) {}

    bool done() const { return done_; }

  protected:
    bool simplified = false;
    Stmt visit(const For &op) override;
};

} // namespace ir

#endif // UNROLL_H
