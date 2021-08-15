#ifndef IR_MULTI_LEVEL_TILING_H
#define IR_MULTI_LEVEL_TILING_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>

#include <array>

namespace ir {
struct MultiLevelTilingAnnotation {
    std::vector<std::array<int, 4>> spaceLoopTiling;
    std::vector<std::array<int, 2>> reductionLoopTiling;
};

class MultiLevelTilingRule : public Rule {
    std::vector<ForsWithDataReuse> targets;

  public:
    int analyze(Schedule &schedule) override;
    SketchPart genPart(int p) override;
};

class MultiLevelTilingPart : public SketchPartNode {
    ForsWithDataReuse target;
    MultiLevelTilingAnnotation annotation;

  public:
    void genRandAnnotation() override;
    explicit MultiLevelTilingPart(ForsWithDataReuse);
    void apply(Schedule &schedule) override;
    SketchPart mutate() override;
    SketchPart crossover(const SketchPart &part) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;
};

} // namespace ir

#endif // IR_MULTI_LEVEL_TILING_H
