#ifndef IR_SKIP_H
#define IR_SKIP_H

#include <auto_schedule/rule.h>
namespace ir {

class SkipRule : public Rule {
  public:
    RuleStatus analyze(const Sketch &sketch) override {
        return RuleStatus::Apply;
    }
    std::vector<Sketch> genPart(const Sketch &sketch) override {
        Sketch newSketch = sketch.clone();
        newSketch.moveToNextTarget();
        return {newSketch};
    };
};
} // namespace ir

#endif
