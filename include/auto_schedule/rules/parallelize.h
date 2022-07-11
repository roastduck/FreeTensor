#ifndef FREE_TENSOR_AUTO_SCHEDULE_PARALLELIZE_H
#define FREE_TENSOR_AUTO_SCHEDULE_PARALLELIZE_H

#include <auto_schedule/rule.h>

namespace freetensor {

class ParallelizeRule : public Rule {

  public:
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class ParallelizePart : public SketchPartNode {
    int maxSize_;
    int parallelSize_;
    ID lastParallelizedID_{};

  public:
    ParallelizePart(size_t maxSize, size_t parallelSize = 0)
        : maxSize_(maxSize), parallelSize_(parallelSize) {}

    void genRandAnnotation(RNG &gen) override;
    void genFakeAnnotation(RNG &gen) override;

    bool mutate(RNG &gen) override;

    bool crossover(const SketchPart &part, RNG &gen) override;

    void apply(Schedule &schedule, SubSketch &subSketch) override;

    SketchPartType partType() override { return SketchPartType::Parallelize; }

    [[nodiscard]] std::vector<int> getAnnotation() const override {
        return {parallelSize_};
    };

    [[nodiscard]] size_t hash() const override {
        return hashCombine(std::hash<std::string>{}("parallelize"),
                           std::hash<int>{}(parallelSize_));
    }

    [[nodiscard]] SketchPart clone() const override {
        return Ref<ParallelizePart>::make(maxSize_, parallelSize_);
    };

    const ID &lastParallelizedID() const { return lastParallelizedID_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_AUTO_SCHEDULE_PARALLELIZE_H
