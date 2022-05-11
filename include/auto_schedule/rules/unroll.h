#ifndef FREE_TENSOR_AUTO_SCHEDULE_UNROLL_H
#define FREE_TENSOR_AUTO_SCHEDULE_UNROLL_H

#include <auto_schedule/rule.h>

namespace freetensor {

class UnrollRule : public Rule {
    TargetType targetType_;

  public:
    UnrollRule(TargetType targetType) : targetType_(targetType) {}
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class UnrollPart : public SketchPartNode {
    TargetType targetType_;
    int maxSize_;

  public:
    UnrollPart(TargetType targetType, size_t max_size = 0)
        : targetType_(targetType), maxSize_(max_size) {}
    void genRandAnnotation(std::default_random_engine &gen) override;
    bool mutate(std::default_random_engine &gen) override;
    bool crossover(const SketchPart &part,
                   std::default_random_engine &gen) override;
    void apply(Schedule &schedule, SketchTarget &target) override;
    SketchPartType partType() override { return SketchPartType::Unroll; }
    [[nodiscard]] std::vector<int> getAnnotation() const override {
        return {maxSize_};
    };
    [[nodiscard]] size_t hash() const override {
        return hashCombine(std::hash<std::string>{}("unroll"),
                           std::hash<int>{}(maxSize_));
    }
    [[nodiscard]] SketchPart clone() const override {
        return Ref<UnrollPart>::make(targetType_, maxSize_);
    };
};

} // namespace freetensor

#endif // FREE_TENSOR_AUTO_SCHEDULE_UNROLL_H
