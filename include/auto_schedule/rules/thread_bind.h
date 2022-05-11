#ifndef FREE_TENSOR_THREAD_BIND_H
#define FREE_TENSOR_THREAD_BIND_H

#include <auto_schedule/rule.h>

namespace freetensor {

class ThreadBindRule : public Rule {
  public:
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class ThreadBindPart : public SketchPartNode {
  public:
    ID lastParallelizedID;
    int vthread_size;
    void genRandAnnotation(std::default_random_engine &gen) override{};
    void apply(Schedule &schedule, SketchTarget &target) override;
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

} // namespace freetensor

#endif // FREE_TENSOR_THREAD_BIND_H
