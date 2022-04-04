#ifndef IR_THREAD_BIND_H
#define IR_THREAD_BIND_H

#include <auto_schedule/rule.h>

namespace ir {

class ThreadBindRule : public InitRule {
  public:
    bool apply(SketchPart &part, Schedule &schedule) const override;
};

}

#endif // IR_THREAD_BIND_H