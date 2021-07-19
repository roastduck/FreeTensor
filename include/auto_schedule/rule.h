#ifndef IR_RULE_H
#define IR_RULE_H

#include <auto_schedule/sketch.h>
#include <schedule.h>

namespace ir {

class Rule {
  public:
    virtual int analyze(Schedule &schedule) = 0;
    virtual SketchPart gen_part(int p) = 0;
};

} // namespace ir

#endif // IR_RULE_H