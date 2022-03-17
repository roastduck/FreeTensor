#ifndef IR_RULE_H
#define IR_RULE_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/sketch.h>
#include <schedule.h>

namespace ir {

enum class RuleStatus { Skip, Apply, ApplyAndSkipRest };

class Rule {
  public:
    virtual RuleStatus analyze(const Sketch &sketch) = 0;
    virtual std::vector<Sketch> genPart(const Sketch &sketch) = 0;
};

} // namespace ir

#endif // IR_RULE_H
