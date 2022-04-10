#include <cmath>
#include <schedule.h>
#include <schedule/multi_level_tiling.h>

namespace ir {

std::vector<std::pair<ID, int>> splitLoop(Schedule &schedule, ID loop,
                                          std::vector<int> tiling) {
    int n = tiling.size();
    std::vector<std::pair<ID, int>> result(n);
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

std::vector<std::pair<ID, int>>
multiLevelTiling(Schedule &schedule, const ForsWithDataReuse &target,
                 const MultiLevelTilingAnnotation &annotation,
                 const std::string &pat) {
    int spaceLoopLength = target.spaceLoops.size();
    int reductionLoopLength = target.reductionLoops.size();
    int spaceLoopTimes = annotation.spaceLoopTiling[0].size();
    int reductionLoopTimes = annotation.reductionLoopTiling.empty()
                                 ? 0
                                 : annotation.reductionLoopTiling[0].size();

    std::vector<std::vector<std::pair<ID, int>>> spaceSplit(spaceLoopLength);
    std::vector<std::vector<std::pair<ID, int>>> reductionSplit(
        reductionLoopLength);

    for (int i = 0; i < spaceLoopLength; i++) {
        spaceSplit[i] = splitLoop(schedule, target.spaceLoops[i].id,
                                  annotation.spaceLoopTiling[i]);
    }
    for (int i = 0; i < reductionLoopLength; i++) {
        reductionSplit[i] = splitLoop(schedule, target.reductionLoops[i].id,
                                      annotation.reductionLoopTiling[i]);
    }

    std::vector<std::pair<ID, int>> tiles;
    tiles.reserve(spaceLoopTimes * spaceLoopLength +
                  reductionLoopTimes * reductionLoopLength);
    int nowSpace = 0;
    int nowReduction = 0;
    for (char c : pat) {
        if (c == 'S') {
            for (auto &split : spaceSplit) {
                tiles.push_back(split[nowSpace]);
            }
            nowSpace++;
        } else {
            for (auto &split : reductionSplit) {
                tiles.push_back(split[nowReduction]);
            }
            nowReduction++;
        }
    }
    std::vector<ID> labels;
    std::cout << "tiles: ";
    for (const auto &tile : tiles) {
        if (tile.second > 1) {
            labels.push_back(tile.first);
            std::cout << tile.first.strId() << " ";
        }
    }
    schedule.reorder(labels);
    return tiles;
}

void multiLevelTilingWithFusion(Schedule &schedule,
                                const ForsWithDataReuse &target,
                                const MultiLevelTilingAnnotation &annotation,
                                const std::string &pat,
                                const ElementWiseInfo &toFuse, int level) {
    auto tiles = multiLevelTiling(schedule, target, annotation, pat);
    std::string fusePat = pat.substr(0, level) + "S";
    std::cout << toString(schedule.ast()) << std::endl;
    std::cout << "Level: " << level << "Fuse Pattern: " << fusePat << std::endl;
    MultiLevelTilingAnnotation fuseAnnotation;
    ForsWithDataReuse fuseTarget;
    for (size_t i = 0; i < toFuse.fors.size(); i++) {
        fuseTarget.spaceLoops = toFuse.fors;
        std::vector<int> tiling(level + 1);
        int tot = 1;
        int tileSize = annotation.spaceLoopTiling[0].size();
        for (int j = 0; j < level; j++) {
            tiling[level - j] = annotation.spaceLoopTiling[i][tileSize - 1 - j];
            tot *= annotation.spaceLoopTiling[i][tileSize - 1 - j];
        }
        tiling[0] = ceil(double(toFuse.fors[i].length) / tot);
        fuseAnnotation.spaceLoopTiling.push_back(tiling);
    }
    auto fuseTiles =
        multiLevelTiling(schedule, fuseTarget, fuseAnnotation, fusePat);
    std::cout << toString(schedule.ast()) << std::endl;
    size_t fuseTileSize = fuseTiles.size() - fuseTarget.spaceLoops.size();
    ID lastFuse;
    std::cout << "before fuse: " << toString(schedule.ast()) << std::endl;
    for (size_t i = 0; i < fuseTileSize; i++) {
        if (fuseTiles[i].second > 1) {
            lastFuse = schedule.fuse(tiles[i].first, fuseTiles[i].first);
        }
    }
    std::cout << "after fuse: " << toString(schedule.ast()) << std::endl;
    schedule.cache(schedule.find(lastFuse).as<ForNode>()->body_->id(),
                   target.dest, MemType::CPU);
    std::cout << "after cache: " << toString(schedule.ast()) << std::endl;
}

} // namespace ir
