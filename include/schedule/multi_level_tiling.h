#ifndef FREE_TENSOR_MULTI_LEVEL_TILING_H
#define FREE_TENSOR_MULTI_LEVEL_TILING_H
#include <auto_schedule/structs.h>

namespace freetensor {
class Schedule;
std::vector<std::pair<ID, int>>
multiLevelTiling(Schedule &schedule, const ForsWithDataReuse &target,
                 const MultiLevelTilingAnnotation &annotation,
                 const std::string &pat);
std::vector<std::pair<ID, int>> multiLevelTilingWithFusion(
    Schedule &schedule, const ForsWithDataReuse &target,
    const MultiLevelTilingAnnotation &annotation, const std::string &pat,
    const ElementWiseInfo &toFuse, int level, MemType memType);
} // namespace freetensor
#endif // FREE_TENSOR_MULTI_LEVEL_TILING_H
