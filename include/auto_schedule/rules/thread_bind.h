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
    ID lastParallelizedID_;
    int vthreadSize_;

  public:
    void genRandAnnotation(RNG &gen) override{};

    void apply(Schedule &schedule, SubSketch &subSketch) override;

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

    const ID &lastParallelizedID() const { return lastParallelizedID_; }
    int vthreadSize() const { return vthreadSize_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_THREAD_BIND_H
