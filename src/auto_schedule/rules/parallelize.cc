#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/parallelize.h>
#include <auto_schedule/utils.h>
#include <schedule/unroll.h>

namespace freetensor {

void ParallelizePart::apply(Schedule &schedule, SketchTarget &target) {
    Ref<MultiLevelTilingPart> part =
        target.getPart(SketchPartType::MultiLevelTiling)
            .as<MultiLevelTilingPart>();
    if (!part.isValid()) {
        part = target.getPart(SketchPartType::MultiLevelTilingWithFusion)
                   .as<MultiLevelTilingPart>();
    }
    std::vector<ID> toFuse;
    auto &tiles = part->tiles();
    int degree = 1;
    for (int i = 0; i < parallelSize_; i++) {
        if (degree * tiles[i].second > MAX_PARALLELIZE) {
            break;
        }
        if (tiles[i].second > 1) {
            toFuse.push_back(tiles[i].first);
            degree *= tiles[i].second;
        }
    }
    if (toFuse.empty()) {
        return;
    }
    lastParallelizedID_ = mergeLoops(schedule, toFuse);
    schedule.parallelize(lastParallelizedID_, OpenMPScope{});
}

void ParallelizePart::genRandAnnotation(std::default_random_engine &gen) {
    parallelSize_ = randomInt(maxSize_ - 1, gen) + 1;
}

void ParallelizePart::genFakeAnnotation(std::default_random_engine &gen) {
    parallelSize_ = maxSize_;
}

bool ParallelizePart::mutate(std::default_random_engine &gen) {
    parallelSize_ = randomInt(maxSize_ - 1, gen) + 1;
    return true;
}
bool ParallelizePart::crossover(const SketchPart &part,
                                std::default_random_engine &gen) {
    if (auto p = part.as<ParallelizePart>();
        p.isValid() && p->partType() == SketchPartType::Parallelize) {
        parallelSize_ = p->parallelSize_;
        return true;
    }
    return false;
}

std::vector<Sketch> ParallelizeRule::genPart(const Sketch &sketch) {
    Sketch newSketch = sketch.clone();
    Ref<MultiLevelTilingPart> part =
        sketch.nowTarget()
            .getPart(SketchPartType::MultiLevelTiling)
            .as<MultiLevelTilingPart>();
    if (!part.isValid()) {
        part = sketch.nowTarget()
                   .getPart(SketchPartType::MultiLevelTilingWithFusion)
                   .as<MultiLevelTilingPart>();
    }
    newSketch.addPart(Ref<ParallelizePart>::make(part->spaceLoopLength() *
                                                 part->frontSpaceLoopTimes()));
    newSketch.addLog("parallelize");
    return {newSketch};
}

RuleStatus ParallelizeRule::analyze(const Sketch &sketch) {
    if (sketch.nowTarget().hasPart(SketchPartType::Parallelize))
        return RuleStatus::Skip;
    if (sketch.nowTarget().hasPart(SketchPartType::MultiLevelTiling) ||
        sketch.nowTarget().hasPart(SketchPartType::MultiLevelTilingWithFusion))
        return RuleStatus::ApplyAndSkipRest;
    return RuleStatus::Skip;
}

} // namespace freetensor
