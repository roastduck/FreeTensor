#ifndef FREE_TENSOR_RULE_H
#define FREE_TENSOR_RULE_H

#include <auto_schedule/sketch.h>
#include <schedule.h>

namespace freetensor {

enum class RuleStatus { Skip, Apply, ApplyAndSkipRest };

class Rule {
  public:
    virtual ~Rule() {}
    virtual RuleStatus analyze(const Sketch &sketch) = 0;
    virtual std::vector<Sketch> genPart(const Sketch &sketch) = 0;
};

} // namespace freetensor

#endif // FREE_TENSOR_RULE_H
