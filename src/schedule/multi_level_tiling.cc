#include <analyze/all_defs.h>
#include <cmath>
#include <schedule.h>
#include <schedule/multi_level_tiling.h>

namespace freetensor {

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

Schedule::IDMap fissionLoops(Schedule &schedule, const std::vector<ID> &loops,
                             ID splitter) {
    int n = loops.size();
    Schedule::IDMap ret;
    for (int i = n - 1; i >= 0; i--) {
        auto now =
            schedule.fission(loops[i], FissionSide::After, splitter).second;
        if (ret.empty()) {
            ret = now;
        } else {
            for (auto &&[key, value] : ret) {
                ret[key] = now[value];
            }
            ret[loops[i]] = loops[i].strId() + ".b";
        }
        splitter = loops[i].strId() + ".a";
    }
    return ret;
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
    //    std::cout << "tiles: ";
    for (const auto &tile : tiles) {
        if (tile.second > 1) {
            labels.push_back(tile.first);
            //            std::cout << tile.first.strId() << " ";
        }
    }
    schedule.reorder(labels);
    return tiles;
}

std::vector<std::pair<ID, int>> multiLevelTilingWithFusion(
    Schedule &schedule, const ForsWithDataReuse &target,
    const MultiLevelTilingAnnotation &annotation, const std::string &pat,
    const ElementWiseInfo &toFuse, int level, TargetType targetType) {
    std::vector<std::pair<ID, int>> tiles;
    std::string firstPat = pat.substr(0, level) + "S";
    if (!target.initStmt.isValid()) {
        tiles = multiLevelTiling(schedule, target, annotation, pat);
    } else {
        ForsWithDataReuse firstTarget;
        MultiLevelTilingAnnotation firstAnnotation;
        firstTarget.spaceLoops = target.spaceLoops;
        int spaceTileSize = annotation.spaceLoopTiling[0].size();
        for (size_t i = 0; i < target.spaceLoops.size(); i++) {
            std::vector<int> tiling(level + 1);
            int tot = 1;
            for (int j = 0; j < level; j++) {
                tiling[level - j] =
                    annotation.spaceLoopTiling[i][spaceTileSize - 1 - j];
                tot *= annotation.spaceLoopTiling[i][spaceTileSize - 1 - j];
            }
            tiling[0] = target.spaceLoops[i].length / tot;
            firstAnnotation.spaceLoopTiling.push_back(tiling);
        }
        auto firstTiles =
            multiLevelTiling(schedule, firstTarget, firstAnnotation, firstPat);
        std::vector<ID> fissionIDs;
        ForsWithDataReuse secondTarget;
        secondTarget.spaceLoops = target.spaceLoops;
        secondTarget.reductionLoops = target.reductionLoops;
        MultiLevelTilingAnnotation secondAnnotation;
        secondAnnotation.reductionLoopTiling = annotation.reductionLoopTiling;
        size_t lastTileStart = firstTiles.size() - target.spaceLoops.size();

        for (size_t i = lastTileStart; i < firstTiles.size(); i++) {
            if (firstTiles[i].second > 1) {
                fissionIDs.push_back(firstTiles[i].first);
                secondTarget.spaceLoops[i - lastTileStart].id =
                    firstTiles[i].first.strId();
                secondTarget.spaceLoops[i - lastTileStart].length =
                    firstTiles[i].second;
            }
        }
        auto mp = fissionLoops(schedule, fissionIDs, target.initStmt);
        for (auto &spaceLoop : secondTarget.spaceLoops) {
            spaceLoop.id = mp[spaceLoop.id];
        }
        for (auto &reductionLoop : secondTarget.reductionLoops) {
            reductionLoop.id = mp[reductionLoop.id];
        }
        auto secondPat = pat.substr(level);
        for (size_t i = 0; i < target.spaceLoops.size(); i++) {
            std::vector<int> tiling(spaceTileSize - level);
            for (int j = 0; j < spaceTileSize - level; j++) {
                tiling[j] = annotation.spaceLoopTiling[i][j];
            }
            secondAnnotation.spaceLoopTiling.push_back(tiling);
        }
        auto secondTiles = multiLevelTiling(schedule, secondTarget,
                                            secondAnnotation, secondPat);
        tiles.insert(tiles.end(), firstTiles.begin(),
                     firstTiles.begin() + lastTileStart);
        tiles.insert(tiles.end(), secondTiles.begin(), secondTiles.end());
    }
    //    std::cout << "Level: " << level << "Fuse Pattern: " << firstPat <<
    //    std::endl;
    MultiLevelTilingAnnotation fuseAnnotation;
    ForsWithDataReuse fuseTarget;
    fuseTarget.spaceLoops = toFuse.fors;
    for (size_t i = 0; i < toFuse.fors.size(); i++) {
        std::vector<int> tiling(level + 1);
        int tot = 1;
        int tileSize = annotation.spaceLoopTiling[0].size();
        for (int j = 0; j < level; j++) {
            tiling[level - j] = annotation.spaceLoopTiling[i][tileSize - 1 - j];
            tot *= annotation.spaceLoopTiling[i][tileSize - 1 - j];
        }
        tiling[0] = toFuse.fors[i].length / tot;
        fuseAnnotation.spaceLoopTiling.push_back(tiling);
    }
    auto fuseTiles =
        multiLevelTiling(schedule, fuseTarget, fuseAnnotation, firstPat);
    //    std::cout << toString(schedule.ast()) << std::endl;
    size_t fuseTileSize = fuseTiles.size() - fuseTarget.spaceLoops.size();
    ID lastFuse;
    //    std::cout << "before fuse: " << toString(schedule.ast()) << std::endl;
    for (size_t i = 0; i < fuseTileSize; i++) {
        if (fuseTiles[i].second > 1) {
            lastFuse = tiles[i].first =
                schedule.fuse(tiles[i].first, fuseTiles[i].first);
        }
    }
    auto body = schedule.find(lastFuse).as<ForNode>()->body_->id();
    try {
        schedule.cache(body, target.dest,
                       targetType == TargetType::CPU ? MemType::CPU
                                                     : MemType::GPULocal);
    } catch (const InvalidSchedule &e) {
    }
    ID firstReduction = lastFuse;
    for (int i = target.reductionLoops.size() - 1; i >= 0; i--) {
        if (tiles[fuseTileSize + i].second > 1) {
            firstReduction = tiles[fuseTileSize + i].first;
            break;
        }
    }
    if (targetType == TargetType::GPU) {
        for (auto &read : target.reads) {
            try {
                body = schedule.find(firstReduction).as<ForNode>()->body_->id();
                schedule.cache(body, read, MemType::GPUShared);
            } catch (const InvalidSchedule &e) {
            }
        }
    }

    return tiles;
}

} // namespace freetensor
