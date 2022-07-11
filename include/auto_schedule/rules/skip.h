#ifndef FREE_TENSOR_SKIP_H
#define FREE_TENSOR_SKIP_H

#include <auto_schedule/rule.h>
namespace freetensor {

class SkipRule : public Rule {
  public:
    RuleStatus analyze(const Sketch &sketch) override {
        return RuleStatus::Apply;
    }
    std::vector<Sketch> genPart(const Sketch &sketch) override {
        Sketch newSketch = sketch.clone();
        newSketch.moveToNextSub();
        return {newSketch};
    };
};
} // namespace freetensor

#endif
