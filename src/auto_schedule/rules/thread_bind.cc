#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/thread_bind.h>

namespace freetensor {

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

void ThreadBindPart::apply(Schedule &schedule, SketchTarget &target) {
    SketchPart part =
        target.getPart(SketchPartType::MultiLevelTilingWithFusion);
    ASSERT(part.isValid());
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
    vthread_size = 1;
    for (size_t i = spaceLoopLength; i < spaceLoopLength * 2; i++) {
        if (tiles[i].second > 1) {
            vthreadLoops.push_back(tiles[i].first);
            vthread_size *= tiles[i].second;
        }
    }
    for (size_t i = spaceLoopLength * 2; i < spaceLoopLength * 3; i++) {
        if (tiles[i].second > 1) {
            threadLoops.push_back(tiles[i].first);
        }
    }
    ID blockID = mergeLoops(schedule, blockLoops);
    ID vthreadID = mergeLoops(schedule, vthreadLoops);
    ID threadID = mergeLoops(schedule, threadLoops);
    if (blockID.isValid()) {
        schedule.parallelize(blockID, blockIdxX);
        lastParallelizedID = blockID;
    }
    if (vthreadID.isValid()) {
        schedule.blend(vthreadID);
    }
    if (threadID.isValid()) {
        schedule.parallelize(threadID, threadIdxX);
        lastParallelizedID = threadID;
    }
}

std::vector<Sketch> ThreadBindRule::genPart(const Sketch &sketch) {
    Sketch newSketch = sketch.clone();
    newSketch.addPart(Ref<ThreadBindPart>::make());
    newSketch.addLog("thread_bind");
    return {newSketch};
}

RuleStatus ThreadBindRule::analyze(const Sketch &sketch) {
    if (sketch.nowTarget().hasPart(SketchPartType::ThreadBind))
        return RuleStatus::Skip;
    if (sketch.nowTarget().hasPart(SketchPartType::MultiLevelTilingWithFusion))
        return RuleStatus::ApplyAndSkipRest;
    return RuleStatus::Skip;
}

} // namespace freetensor
