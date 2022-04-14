#include <analyze/find_elementwise.h>
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/utils.h>
#include <unistd.h>

#include <utility>

namespace ir {
RuleStatus MultiLevelTilingWithFusionRule::analyze(const Sketch &sketch) {
    if (sketch.nowTarget().hasPart(
            SketchPartType::MultiLevelTilingWithFusion) ||
        sketch.nowTarget().hasPart(SketchPartType::MultiLevelTiling)) {
        return RuleStatus::Skip;
    }
    if (auto toFuse = findSingleElementWiseConsumer(sketch.schedule().ast(),
                                                    sketch.nowTarget().target);
        toFuse.isValid()) {
        toFuse_ = toFuse;
        return memType_ == MemType::CPU ? RuleStatus::Apply
                                        : RuleStatus::ApplyAndSkipRest;
    }
    return RuleStatus::Skip;
}

std::vector<Sketch>
MultiLevelTilingWithFusionRule::genPart(const Sketch &sketch) {
    std::vector<Sketch> ret;
    for (size_t i = 0; i < fuseLevels_.size(); i++) {
        Sketch newSketch = sketch.clone();
        newSketch.addPart(Ref<MultiLevelTilingWithFusionPart>::make(
            sketch.nowTarget().target, toFuse_, fuseLevels_[i], pat_,
            memType_));
        newSketch.addLog("multi_level_tiling_with_fusion " +
                         std::to_string(fuseLevels_[i]));
        ret.push_back(newSketch);
    }
    return ret;
}

void MultiLevelTilingWithFusionPart::genRandAnnotation(
    std::default_random_engine &gen) {
    int spaceLoopLength = target_.spaceLoops.size();
    int reductionLoopLength = target_.reductionLoops.size();
    std::vector<std::vector<int>> spaceLoopTiling(spaceLoopLength);
    std::vector<std::vector<int>> reductionLoopTiling(reductionLoopLength);
    while (true) {
        int vthread = 1;
        int thread = 1;
        for (int i = 0; i < spaceLoopLength; i++) {
            spaceLoopTiling[i] = randomFillArray(target_.spaceLoops[i].length,
                                                 spaceLoopTimes_, gen);
            vthread *= spaceLoopTiling[i][spaceLoopTimes_ - 2];
            thread *= spaceLoopTiling[i][spaceLoopTimes_ - 3];
        }
        if (vthread <= 16 && thread <= 1024) {
            break;
        }
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionLoopTiling[i] = randomFillArray(
            target_.reductionLoops[i].length, reductionLoopTimes_, gen);
    }
    annotation_.spaceLoopTiling = spaceLoopTiling;
    annotation_.reductionLoopTiling = reductionLoopTiling;
}

MultiLevelTilingWithFusionPart::MultiLevelTilingWithFusionPart(
    ForsWithDataReuse fors, ElementWiseInfo toFuse, int level, std::string pat,
    MemType memType)
    : MultiLevelTilingPart(std::move(fors), std::move(pat)), memType_(memType),
      level_(level), toFuse_(std::move(toFuse)) {}

void MultiLevelTilingWithFusionPart::apply(Schedule &schedule,
                                           SketchTarget &target) {
    tiles_ = schedule.multiLevelTilingWithFusion(target_, annotation_, pat_,
                                                 toFuse_, level_, memType_);
}

bool MultiLevelTilingWithFusionPart::mutate(std::default_random_engine &gen) {
    // std::cout << "Start mutating...\n";
    MultiLevelTilingAnnotation mut = annotation_;
    int mutPart = randomInt(1, gen);
    int spaceSize = target_.spaceLoops.size();
    int reduceSize = target_.reductionLoops.size();
    if (!spaceSize) {
        mutPart = 1;
    } else if (!reduceSize) {
        mutPart = 0;
    }
    if (mutPart == 0) {
        int mut_idx = randomInt(target_.spaceLoops.size() - 1, gen);
        mut.spaceLoopTiling[mut_idx] = randomFillArray(
            target_.spaceLoops[mut_idx].length, spaceLoopTimes_, gen);
        int vthread = 1;
        int thread = 1;
        for (size_t i = 0; i < target_.spaceLoops.size(); i++) {
            vthread *= mut.spaceLoopTiling[i][spaceLoopTimes_ - 2];
            thread *= mut.spaceLoopTiling[i][spaceLoopTimes_ - 3];
        }
        if (vthread > 16 || thread > 1024) {
            return false;
        }
    } else {
        int mut_idx = randomInt(target_.reductionLoops.size() - 1, gen);
        mut.reductionLoopTiling[mut_idx] = randomFillArray(
            target_.reductionLoops[mut_idx].length, reductionLoopTimes_, gen);
    }
    annotation_ = mut;
    // std::cout << "End mutating...\n";
    return true;
}

bool MultiLevelTilingWithFusionPart::crossover(
    const SketchPart &part, std::default_random_engine &gen) {
    // std::cout << "Start crossover...\n";
    if (part->partType() != SketchPartType::MultiLevelTilingWithFusion)
        return false;
    auto p = part.as<MultiLevelTilingWithFusionPart>();
    MultiLevelTilingAnnotation mut = annotation_;
    int mutPart = randomInt(1, gen);
    int spaceSize = target_.spaceLoops.size();
    int reduceSize = target_.reductionLoops.size();
    if (!spaceSize) {
        mutPart = 1;
    } else if (!reduceSize) {
        mutPart = 0;
    }
    if (mutPart == 0) {
        int mutIdx = randomInt(target_.spaceLoops.size() - 1, gen);
        mut.spaceLoopTiling[mutIdx] = p->annotation_.spaceLoopTiling[mutIdx];
        int vthread = 1;
        int thread = 1;
        for (size_t i = 0; i < target_.spaceLoops.size(); i++) {
            vthread *= mut.spaceLoopTiling[i][spaceLoopTimes_ - 2];
            thread *= mut.spaceLoopTiling[i][spaceLoopTimes_ - 3];
        }
        if (vthread > 16 || thread > 1024) {
            return false;
        }
    } else {
        int mutIdx = randomInt(target_.reductionLoops.size() - 1, gen);
        mut.reductionLoopTiling[mutIdx] =
            p->annotation_.reductionLoopTiling[mutIdx];
    }
    annotation_ = mut;
    // std::cout << "End crossover...\n";
    return true;
}

std::vector<int> MultiLevelTilingWithFusionPart::getAnnotation() const {
    std::vector<int> ret;
    for (auto &item : annotation_.spaceLoopTiling) {
        ret.insert(ret.end(), item.begin(), item.end());
    }
    for (auto &item : annotation_.reductionLoopTiling) {
        ret.insert(ret.end(), item.begin(), item.end());
    }
    // std::cout << "Annotation: ";
    // for (int item : ret) {
    //     std::cout << item << " ";
    // }
    // std::cout << "\n";
    return ret;
}

size_t MultiLevelTilingWithFusionPart::hash() const {
    size_t h = std::hash<ForsWithDataReuse>{}(target_);
    h = hashCombine(h, std::hash<MultiLevelTilingAnnotation>{}(annotation_));
    h = hashCombine(h, std::hash<ElementWiseInfo>{}(toFuse_));
    h = hashCombine(h, std::hash<int>{}(level_));
    return h;
}

} // namespace ir
