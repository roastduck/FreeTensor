#include <analyze/find_elementwise.h>
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/cache_write.h>

namespace freetensor {
RuleStatus CacheWriteRule::analyze(const Sketch &sketch) {
    if (!findSingleElementWiseConsumer(sketch.schedule().ast(),
                                       sketch.nowSubSketch().target)
             .isValid()) {
        return memType_ == MemType::CPU ? RuleStatus::Apply
                                        : RuleStatus::ApplyAndSkipRest;
    }
    return RuleStatus::Skip;
}

std::vector<Ref<Sketch>> CacheWriteRule::genPart(const Sketch &sketch) {
    auto newSketch = sketch.clone();
    auto &target = newSketch->nowSubSketch().target;
    if (verbose_ >= 2) {
        logger() << "cache: " << target.outermost << " " << target.dest
                 << std::endl;
    }
    std::string name = std::get<2>(
        newSketch->schedule().cache(target.outermost, target.dest, memType_));
    target.dest = name;
    newSketch->addLog("cache_write_" +
                      std::to_string(std::hash<ForsWithDataReuse>{}(target)));
    return {newSketch};
}

} // namespace freetensor
