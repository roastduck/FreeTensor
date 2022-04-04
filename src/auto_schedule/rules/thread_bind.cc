#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/thread_bind.h>

namespace ir {

ID mergeLoops(Schedule &schedule, std::vector<ID> loops) {
    if (loops.empty()) {
        return {};
    }
    ID outermost = loops[0];
    for (size_t i = 1; i < loops.size(); i++) {
        outermost = schedule.merge(outermost, loops[i]);
    }
    return outermost;
}

bool ThreadBindRule::apply(SketchPart &part, Schedule &schedule) const {
    std::cout << "begin thread bind" << std::endl;
    if (part->partType() != SketchPartType::MultiLevelTilingWithFusion &&
        part->partType() != SketchPartType::MultiLevelTiling) {
        return false;
    }
    auto mltPart = part.as<MultiLevelTilingPart>();
    auto &tiles = mltPart->tiles();
    size_t spaceLoopLength = mltPart->spaceLoopLength();
    std::vector<ID> blockLoops;
    std::vector<ID> vthreadLoops;
    std::vector<ID> threadLoops;
    for (size_t i = 0; i < spaceLoopLength; i++) {
        if (tiles[i].second > 1) {
            blockLoops.push_back(tiles[i].first);
        }
    }
    for (size_t i = spaceLoopLength; i < spaceLoopLength * 2; i++) {
        if (tiles[i].second > 1) {
            vthreadLoops.push_back(tiles[i].first);
        }
    }
    for (size_t i = spaceLoopLength * 2; i < spaceLoopLength * 3; i++) {
        if (tiles[i].second > 1) {
            threadLoops.push_back(tiles[i].first);
        }
    }
    std::cout << "before merge: " << toString(schedule.ast()) << std::endl;
    ID blockID = mergeLoops(schedule, blockLoops);
    std::cout << "after block merge: " << toString(schedule.ast()) << std::endl;
    ID vthreadID = mergeLoops(schedule, vthreadLoops);
    std::cout << "after vthread merge: " << toString(schedule.ast())
              << std::endl;
    ID threadID = mergeLoops(schedule, threadLoops);
    std::cout << "after thread merge: " << toString(schedule.ast())
              << std::endl;
    if (blockID.isValid()) {
        schedule.parallelize(blockID, blockIdxX);
    }
    std::cout << "after block: " << toString(schedule.ast()) << std::endl;
    if (vthreadID.isValid()) {
        schedule.blend(vthreadID);
    }
    std::cout << "after blend: " << toString(schedule.ast()) << std::endl;
    if (threadID.isValid()) {
        schedule.parallelize(threadID, threadIdxX);
    }
    std::cout << "after thread: " << toString(schedule.ast()) << std::endl;
    std::cout << "thread bind ended" << std::endl;
    return true;
}

} // namespace ir