#include <analyze/find_elementwise.h>
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/utils.h>

#include <utility>

namespace ir {
RuleStatus MultiLevelTilingWithFusionRule::analyze(const Sketch &sketch) {
    if (auto toFuse = findSingleElementWiseConsumer(sketch.schedule().ast(),
                                                    sketch.nowTarget());
        toFuse.isValid()) {
        toFuse_ = toFuse;
        return RuleStatus::Apply;
    }
    return RuleStatus::Skip;
}

std::vector<Sketch>
MultiLevelTilingWithFusionRule::genPart(const Sketch &sketch) {
    std::vector<int> fuseLevel = {1, 2};
    std::vector<Sketch> ret;
    for (size_t i = 0; i < fuseLevel.size(); i++) {
        Sketch newSketch = sketch;
        newSketch.addPart(new MultiLevelTilingWithFusionPart(
            sketch.nowTarget(), toFuse_, fuseLevel[i], "SSRSRS"));
        newSketch.moveToNextTarget();
        ret.push_back(std::move(newSketch));
    }
    return ret;
}

void MultiLevelTilingWithFusionPart::genRandAnnotation(
    std::default_random_engine &gen) {
    int spaceLoopLength = target_.spaceLoops.size();
    int reductionLoopLength = target_.reductionLoops.size();
    std::vector<std::vector<int>> spaceLoopTiling(spaceLoopLength);
    std::vector<std::vector<int>> reductionLoopTiling(reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        spaceLoopTiling[i] =
            randomFillArray(target_.spaceLoops[i].length, spaceLoopTimes_, gen);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionLoopTiling[i] = randomFillArray(
            target_.reductionLoops[i].length, reductionLoopTimes_, gen);
    }
    annotation_.spaceLoopTiling = spaceLoopTiling;
    annotation_.reductionLoopTiling = reductionLoopTiling;
}

MultiLevelTilingWithFusionPart::MultiLevelTilingWithFusionPart(
    ForsWithDataReuse fors, ElementWiseInfo toFuse, int level, std::string pat)
    : pat_(std::move(pat)), level_(level), toFuse_(std::move(toFuse)) {
    target_ = std::move(fors);
    spaceLoopTimes_ = 0;
    reductionLoopTimes_ = 0;
    for (auto c : pat_) {
        if (c == 'S') {
            spaceLoopTimes_++;
        } else {
            reductionLoopTimes_++;
        }
    }
}

void MultiLevelTilingWithFusionPart::apply(Schedule &schedule) {
    schedule.multiLevelTilingWithFusion(target_, annotation_, pat_, toFuse_,
                                        level_);
}

SketchPart
MultiLevelTilingWithFusionPart::mutate(std::default_random_engine &gen) {
    // std::cout << "Start mutating...\n";
    MultiLevelTilingWithFusionPart mut = *this;
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
        mut.annotation_.spaceLoopTiling[mut_idx] = randomFillArray(
            target_.spaceLoops[mut_idx].length, spaceLoopTimes_, gen);

    } else {
        int mut_idx = randomInt(target_.reductionLoops.size() - 1, gen);
        mut.annotation_.reductionLoopTiling[mut_idx] = randomFillArray(
            target_.reductionLoops[mut_idx].length, reductionLoopTimes_, gen);
    }
    // std::cout << "End mutating...\n";
    return Ref<MultiLevelTilingWithFusionPart>::make(std::move(mut));
}

SketchPart
MultiLevelTilingWithFusionPart::crossover(const SketchPart &part,
                                          std::default_random_engine &gen) {
    // std::cout << "Start crossover...\n";
    if (part->partType() != SketchPartType::MultiLevelTilingWithFusion)
        return nullptr;
    auto p = part.as<MultiLevelTilingWithFusionPart>();
    MultiLevelTilingWithFusionPart mut = *this;
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
        mut.annotation_.spaceLoopTiling[mutIdx] =
            p->annotation_.spaceLoopTiling[mutIdx];
    } else {
        int mutIdx = randomInt(target_.reductionLoops.size() - 1, gen);
        mut.annotation_.reductionLoopTiling[mutIdx] =
            p->annotation_.reductionLoopTiling[mutIdx];
    }
    // std::cout << "End crossover...\n";
    return Ref<MultiLevelTilingWithFusionPart>::make(std::move(mut));
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
void MultiLevelTilingWithFusionPart::genAverageAnnotation() {
    int spaceLoopLength = target_.spaceLoops.size();
    int reductionLoopLength = target_.reductionLoops.size();
    std::vector<std::vector<int>> spaceLoopTiling(spaceLoopLength);
    std::vector<std::vector<int>> reductionLoopTiling(reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        int len = target_.spaceLoops[i].length;
        int div = floor(pow(len, 1. / spaceLoopTimes_));
        int last = ceil(double(len) / pow(div, spaceLoopTimes_ - 1));

        spaceLoopTiling[i] = std::vector<int>(spaceLoopTimes_, div);
        spaceLoopTiling[i][spaceLoopTimes_ - 1] = last;
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        int len = target_.reductionLoops[i].length;
        int div = floor(pow(len, 1. / reductionLoopTimes_));
        int last = ceil(double(len) / pow(div, reductionLoopTimes_ - 1));
        reductionLoopTiling[i] = std::vector<int>(reductionLoopTimes_, div);
        reductionLoopTiling[i][reductionLoopTimes_ - 1] = last;
    }
    annotation_.spaceLoopTiling = spaceLoopTiling;
    annotation_.reductionLoopTiling = reductionLoopTiling;
}

} // namespace ir
