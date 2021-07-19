#ifndef IR_MULTI_LEVEL_TILING_H
#define IR_MULTI_LEVEL_TILING_H

#include <auto_schedule/analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>

#include <array>

namespace ir {
struct MultiLevelTilingAnnotation {
    std::array<int, 4> i_tiling;
    std::array<int, 4> j_tiling;
    std::array<int, 2> k_tiling;
};

class MultiLevelTilingRule : public Rule {
    std::vector<ThreeNestedFors> targets;

  public:
    int analyze(Schedule &schedule) override;
    SketchPart gen_part(int p) override;
};

class MultiLevelTilingPart : public SketchPartNode {
    ThreeNestedFors target;
    MultiLevelTilingAnnotation annotation;

  public:
    void gen_rand_annotation() override;
    explicit MultiLevelTilingPart(ThreeNestedFors);
    void apply(Schedule &schedule) override;
    SketchPart mutate() override;
    SketchPart crossover(const SketchPart &part) override;
    [[nodiscard]] std::vector<int> get_annotation() const override;
};

} // namespace ir

#endif // IR_MULTI_LEVEL_TILING_H