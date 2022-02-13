// #include <analyze/find_all_loops.h>
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>

namespace ir {
int MultiLevelTilingRule::analyze(Schedule &schedule) {
    targets = findMultiLevelTiling(schedule.ast());
    if (targets.empty())
        return false;
    return true;
}

SketchPart MultiLevelTilingRule::genPart(int p) {
    return SketchPart(new MultiLevelTilingPart(targets[p]));
}

void MultiLevelTilingPart::genRandAnnotation(std::default_random_engine &gen) {
    int spaceLoopLength = target.spaceLoops.size();
    int reductionLoopLength = target.reductionLoops.size();
    std::vector<std::array<int, 4>> spaceLoopTiling(spaceLoopLength);
    std::vector<std::array<int, 2>> reductionLoopTiling(reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        spaceLoopTiling[i] =
            randomFillArray<4>(target.spaceLoops[i].length, gen);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionLoopTiling[i] =
            randomFillArray<2>(target.reductionLoops[i].length, gen);
    }
    annotation.spaceLoopTiling = spaceLoopTiling;
    annotation.reductionLoopTiling = reductionLoopTiling;
}

MultiLevelTilingPart::MultiLevelTilingPart(ForsWithDataReuse fors) {
    target = std::move(fors);
}

template <int n>
std::array<std::pair<ID, int>, n> splitLoop(Schedule &schedule, ID loop,
                                            std::array<int, n> tiling) {
    std::array<std::pair<ID, int>, n> result;
    for (int i = 0; i < n - 1; i++) {
        if (tiling[i] != 1) {
            auto t = schedule.split(loop, tiling[i]);
            loop = t.first;
            result[n - i - 1] = std::make_pair(t.second, tiling[i]);
        } else {
            result[n - i - 1] = std::make_pair("", 1);
        }
    }
    result[0] = std::make_pair(loop, tiling[n - 1]);
    return result;
}

void MultiLevelTilingPart::apply(Schedule &schedule) {
    int spaceLoopLength = target.spaceLoops.size();
    int reductionLoopLength = target.reductionLoops.size();

    std::vector<std::array<std::pair<ID, int>, 4>> spaceSplit(spaceLoopLength);
    std::vector<std::array<std::pair<ID, int>, 2>> reductionSplit(
        spaceLoopLength);

    for (int i = 0; i < spaceLoopLength; i++) {
        spaceSplit[i] = splitLoop<4>(schedule, target.spaceLoops[i].id,
                                      annotation.spaceLoopTiling[i]);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionSplit[i] = splitLoop<2>(schedule, target.reductionLoops[i].id,
                                          annotation.reductionLoopTiling[i]);
    }

    std::vector<std::pair<ID, int>> tiles(4 * spaceLoopLength +
                                          2 * reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        tiles[i] = spaceSplit[i][0];
        tiles[i + spaceLoopLength] = spaceSplit[i][1];
        tiles[i + 2 * spaceLoopLength + reductionLoopLength] = spaceSplit[i][2];
        tiles[i + 3 * spaceLoopLength + 2 * reductionLoopLength] =
            spaceSplit[i][3];
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        tiles[i + 2 * spaceLoopLength] = reductionSplit[i][0];
        tiles[i + 3 * spaceLoopLength + reductionLoopLength] =
            reductionSplit[i][1];
    }
    std::vector<ID> labels;
    for (const auto &tile : tiles) {
        if (tile.second > 1) {
            labels.push_back(tile.first);
        }
    }
    schedule.reorder(labels);
}

SketchPart MultiLevelTilingPart::mutate(std::default_random_engine &gen) {
    // std::cout << "Start mutating...\n";
    MultiLevelTilingPart mut = *this;
    int mutPart = randomInt(1, gen);
    int spaceSize = target.spaceLoops.size();
    int reduceSize = target.reductionLoops.size();
    if (!spaceSize) {
        mutPart = 1;
    } else if (!reduceSize) {
        mutPart = 0;
    }
    if (mutPart == 0) {
        int mut_idx = randomInt(target.spaceLoops.size() - 1, gen);
        mut.annotation.spaceLoopTiling[mut_idx] =
            randomFillArray<4>(target.spaceLoops[mut_idx].length, gen);

    } else {
        int mut_idx = randomInt(target.reductionLoops.size() - 1, gen);
        mut.annotation.reductionLoopTiling[mut_idx] =
            randomFillArray<2>(target.reductionLoops[mut_idx].length, gen);
    }
    // std::cout << "End mutating...\n";
    return Ref<MultiLevelTilingPart>::make(std::move(mut));
}

SketchPart MultiLevelTilingPart::crossover(const SketchPart &part,
                                           std::default_random_engine &gen) {
    // std::cout << "Start crossover...\n";
    if (typeid(*(part.get())) != typeid(MultiLevelTilingPart))
        return nullptr;
    auto p = part.as<MultiLevelTilingPart>();
    MultiLevelTilingPart mut = *this;
    int mutPart = randomInt(1, gen);
    int spaceSize = target.spaceLoops.size();
    int reduceSize = target.reductionLoops.size();
    if (!spaceSize) {
        mutPart = 1;
    } else if (!reduceSize) {
        mutPart = 0;
    }
    if (mutPart == 0) {
        int mutIdx = randomInt(target.spaceLoops.size() - 1, gen);
        mut.annotation.spaceLoopTiling[mutIdx] =
            p->annotation.spaceLoopTiling[mutIdx];
    } else {
        int mutIdx = randomInt(target.reductionLoops.size() - 1, gen);
        mut.annotation.reductionLoopTiling[mutIdx] =
            p->annotation.reductionLoopTiling[mutIdx];
    }
    // std::cout << "End crossover...\n";
    return Ref<MultiLevelTilingPart>::make(std::move(mut));
}

std::vector<int> MultiLevelTilingPart::getAnnotation() const {
    std::vector<int> ret;
    for (auto &item : annotation.spaceLoopTiling) {
        ret.insert(ret.end(), item.begin(), item.end());
    }
    for (auto &item : annotation.reductionLoopTiling) {
        ret.insert(ret.end(), item.begin(), item.end());
    }
    // std::cout << "Annotation: ";
    // for (int item : ret) {
    //     std::cout << item << " ";
    // }
    // std::cout << "\n";
    return ret;
}

size_t MultiLevelTilingPart::hash() const {
    size_t h = std::hash<ForsWithDataReuse>{}(target);
    h = hashCombine(h, std::hash<MultiLevelTilingAnnotation>{}(annotation));
    return h;
}

} // namespace ir
