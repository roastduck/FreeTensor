#ifndef UNROLL_H
#define UNROLL_H

#include <mutator.h>
#include <schedule.h>

namespace ir {

class Unroll : public Mutator {
    friend class Schedule;

    std::string loop_;
	unsigned int unroll_num_;
    bool done_ = false;

  public:
    Unroll(const std::string &loop, const unsigned int unroll_num = 0)
        : loop_(loop), unroll_num_(unroll_num) {}

    bool done() const { return done_; }

  protected:
	bool work = false;
    Stmt visit(const For &op) override;
};

} // namespace ir

#endif // UNROLL_H
