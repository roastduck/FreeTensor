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

void MultiLevelTilingPart::genRandAnnotation(std::mt19937 gen) {
    int spaceLoopLength = target.spaceLoops.size();
    int reductionLoopLength = target.reductionLoops.size();
    std::vector<std::array<int, 4>> spaceLoopTiling(spaceLoopLength);
    std::vector<std::array<int, 2>> reductionLoopTiling(reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        spaceLoopTiling[i] = random_fill_array<4>(target.spaceLoops[i].length, gen);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionLoopTiling[i] =
            random_fill_array<2>(target.reductionLoops[i].length, gen);
    }
    annotation.spaceLoopTiling = spaceLoopTiling;
    annotation.reductionLoopTiling = reductionLoopTiling;
}

MultiLevelTilingPart::MultiLevelTilingPart(ForsWithDataReuse fors) {
    target = std::move(fors);
}

template <int n>
std::array<std::pair<std::string, int>, n>
split_loop(Schedule &schedule, std::string loop, std::array<int, n> tiling) {
    std::array<std::pair<std::string, int>, n> result;
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

    std::vector<std::array<std::pair<std::string, int>, 4>> space_split(
        spaceLoopLength);
    std::vector<std::array<std::pair<std::string, int>, 2>> reduction_split(
        spaceLoopLength);

    for (int i = 0; i < spaceLoopLength; i++) {
        space_split[i] = split_loop<4>(schedule, target.spaceLoops[i].id,
                                       annotation.spaceLoopTiling[i]);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reduction_split[i] =
            split_loop<2>(schedule, target.reductionLoops[i].id,
                          annotation.reductionLoopTiling[i]);
    }

    std::vector<std::pair<std::string, int>> tiles(4 * spaceLoopLength +
                                                   2 * reductionLoopLength);
    for (int i = 0; i < spaceLoopLength; i++) {
        tiles[i] = space_split[i][0];
        tiles[i + spaceLoopLength] = space_split[i][1];
        tiles[i + 2 * spaceLoopLength + reductionLoopLength] =
            space_split[i][2];
        tiles[i + 3 * spaceLoopLength + 2 * reductionLoopLength] =
            space_split[i][3];
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        tiles[i + 2 * spaceLoopLength] = reduction_split[i][0];
        tiles[i + 3 * spaceLoopLength + reductionLoopLength] =
            reduction_split[i][1];
    }
    std::vector<std::string> labels;
    for (const auto &tile : tiles) {
        if (tile.second > 1) {
            labels.push_back(tile.first);
        }
    }
    schedule.reorder(labels);
}

SketchPart MultiLevelTilingPart::mutate(std::mt19937 &gen) {
    // std::cout << "Start mutating...\n";
    MultiLevelTilingPart mut = *this;
    int mut_part = random_int(1, gen);
    int spaceSize = target.spaceLoops.size();
    int reduceSize = target.reductionLoops.size();
    if (!spaceSize) {
        mut_part = 1;
    } else if (!reduceSize) {
        mut_part = 0;
    }
    if (mut_part == 0) {
        int mut_idx = random_int(target.spaceLoops.size() - 1, gen);
        mut.annotation.spaceLoopTiling[mut_idx] =
            random_fill_array<4>(target.spaceLoops[mut_idx].length, gen);

    } else {
        int mut_idx = random_int(target.reductionLoops.size() - 1, gen);
        mut.annotation.reductionLoopTiling[mut_idx] =
            random_fill_array<2>(target.reductionLoops[mut_idx].length, gen);
    }
    // std::cout << "End mutating...\n";
    return Ref<MultiLevelTilingPart>::make(std::move(mut));
}

SketchPart MultiLevelTilingPart::crossover(const SketchPart &part, std::mt19937 &gen) {
    // std::cout << "Start crossover...\n";
    if (typeid(*(part.get())) != typeid(MultiLevelTilingPart))
        return nullptr;
    auto p = part.as<MultiLevelTilingPart>();
    MultiLevelTilingPart mut = *this;
    int mut_part = random_int(1, gen);
    int spaceSize = target.spaceLoops.size();
    int reduceSize = target.reductionLoops.size();
    if (!spaceSize) {
        mut_part = 1;
    } else if (!reduceSize) {
        mut_part = 0;
    }
    if (mut_part == 0) {
        int mut_idx = random_int(target.spaceLoops.size() - 1, gen);
        mut.annotation.spaceLoopTiling[mut_idx] =
            p->annotation.spaceLoopTiling[mut_idx];
    } else {
        int mut_idx = random_int(target.reductionLoops.size() - 1, gen);
        mut.annotation.reductionLoopTiling[mut_idx] =
            p->annotation.reductionLoopTiling[mut_idx];
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
    boost::hash_combine(h, std::hash<MultiLevelTilingAnnotation>{}(annotation));
    return h;
}

} // namespace ir
