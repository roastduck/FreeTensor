#ifndef IR_THREAD_BIND_H
#define IR_THREAD_BIND_H

#include <auto_schedule/rule.h>

namespace ir {

class ThreadBindRule : public Rule {
  public:
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class ThreadBindPart : public SketchPartNode {
    void genRandAnnotation(std::default_random_engine &gen) override{};
    void apply(Schedule &schedule, struct SketchTarget &target) override;
    SketchPartType partType() override { return SketchPartType::ThreadBind; }
    [[nodiscard]] std::vector<int> getAnnotation() const override {
        return {};
    };
    [[nodiscard]] size_t hash() const override {
        return std::hash<std::string>{}("thread bind");
    }
    [[nodiscard]] SketchPart clone() const override {
        return Ref<ThreadBindPart>::make();
    };
};

} // namespace ir

#endif // IR_THREAD_BIND_H
