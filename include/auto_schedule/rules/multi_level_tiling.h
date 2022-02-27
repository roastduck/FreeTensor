#ifndef IR_MULTI_LEVEL_TILING_H
#define IR_MULTI_LEVEL_TILING_H

#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>

#include <array>
#include <random>

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
    void genRandAnnotation(std::default_random_engine &gen) override;
    explicit MultiLevelTilingPart(ForsWithDataReuse);
    void apply(Schedule &schedule) override;
    SketchPart mutate(std::default_random_engine &gen) override;
    SketchPart crossover(const SketchPart &part,
                         std::default_random_engine &gen) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;
    [[nodiscard]] size_t hash() const override;
    SketchPartType partType() override {
        return SketchPartType::MultiLevelTiling;
    }
};

} // namespace ir

template <> struct std::hash<ir::MultiLevelTilingAnnotation> {
    std::size_t
    operator()(ir::MultiLevelTilingAnnotation const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &t : s.spaceLoopTiling)
            for (const auto i : t)
                h = ir::hashCombine(h, std::hash<int>{}(i));
        for (const auto &t : s.reductionLoopTiling)
            for (const auto i : t)
                h = ir::hashCombine(h, std::hash<int>{}(i));
        return h;
    }
};

#endif // IR_MULTI_LEVEL_TILING_H
