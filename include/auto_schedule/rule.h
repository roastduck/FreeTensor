#ifndef IR_RULE_H
#define IR_RULE_H

#include <auto_schedule/sketch.h>
#include <schedule.h>

namespace ir {

/**
 * A subclass of Rule must meet the following conditions:
 * 1. The result of analyzation can only rely on the structure of the AST,
 *    but never the specific parameter in the AST.
 */

class Rule {
  public:
    virtual int analyze(Schedule &schedule) = 0;
    virtual SketchPart genPart(int p) = 0;
    virtual ~Rule() {}
};

} // namespace ir

#endif // IR_RULE_H
