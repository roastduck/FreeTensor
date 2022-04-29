#ifndef FREE_TENSOR_MULTI_LEVEL_TILING_H
#define FREE_TENSOR_MULTI_LEVEL_TILING_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/structs.h>

#include <array>
#include <random>

namespace freetensor {

class MultiLevelTilingRule : public Rule {
    std::string pat_;

  public:
    explicit MultiLevelTilingRule(TargetType target) {
        if (target == TargetType::CPU) {
            pat_ = "SSRSRS";
        } else {
            pat_ = "SSSRRSRS";
        }
    }
    RuleStatus analyze(const Sketch &sketch) override;
    std::vector<Sketch> genPart(const Sketch &sketch) override;
};

class MultiLevelTilingPart : public SketchPartNode {
  protected:
    ForsWithDataReuse target_;
    MultiLevelTilingAnnotation annotation_;
    std::string pat_;
    int spaceLoopTimes_;
    int reductionLoopTimes_;
    std::vector<std::pair<ID, int>> tiles_;

  public:
    void genRandAnnotation(std::default_random_engine &gen) override;
    void genSampleAnnotation();
    std::vector<std::pair<ID, int>> &tiles() { return tiles_; }
    explicit MultiLevelTilingPart(ForsWithDataReuse fors,
                                  std::string pat = "SSRSRS");
    void apply(Schedule &schedule, SketchTarget &target) override;
    bool mutate(std::default_random_engine &gen) override;
    bool crossover(const SketchPart &part,
                   std::default_random_engine &gen) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;
    size_t spaceLoopLength() const { return target_.spaceLoops.size(); }
    [[nodiscard]] size_t hash() const override;
    SketchPartType partType() override {
        return SketchPartType::MultiLevelTiling;
    }
    [[nodiscard]] SketchPart clone() const override {
        return Ref<MultiLevelTilingPart>::make(*this);
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_MULTI_LEVEL_TILING_H
