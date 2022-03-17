#include <analyze/find_elementwise.h>
#include <analyze/find_multi_level_tiling.h>
#include <auto_schedule/rules/cache_write.h>

namespace ir {
RuleStatus CacheWriteRule::analyze(const Sketch &sketch) {
    if (!findSingleElementWiseConsumer(sketch.schedule().ast(),
                                       sketch.nowTarget().dest)
             .isValid()) {
        return RuleStatus::Apply;
    }
    return RuleStatus::Skip;
}

std::vector<Sketch> CacheWriteRule::genPart(const Sketch &sketch) {
    Sketch newSketch = sketch;
    const auto &target = newSketch.nowTarget();
    std::cout << "before: " << toString(newSketch.schedule().ast())
              << std::endl;
    std::string name = std::get<2>(newSketch.schedule().cache(
        target.outermost, target.dest, MemType::CPU)); // fixme: GPU Memtype
    std::cout << "after: " << toString(newSketch.schedule().ast()) << std::endl;
    newSketch.nowTarget().dest = name;
    newSketch.addDoneRule(
        "cache_write_" +
        std::to_string(std::hash<ForsWithDataReuse>{}(target)));
    return {newSketch};
}

} // namespace ir
