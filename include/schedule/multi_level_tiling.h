#ifndef IR_MULTI_LEVEL_TILING_H
#define IR_MULTI_LEVEL_TILING_H
#include <auto_schedule/structs.h>

namespace ir {
class Schedule;
std::vector<std::pair<ID, int>>
multiLevelTiling(Schedule &schedule, const ForsWithDataReuse &target,
                 const MultiLevelTilingAnnotation &annotation,
                 const std::string &pat);
void multiLevelTilingWithFusion(Schedule &schedule,
                                const ForsWithDataReuse &target,
                                const MultiLevelTilingAnnotation &annotation,
                                const std::string &pat,
                                const ElementWiseInfo &toFuse, int level);
} // namespace ir
#endif // IR_MULTI_LEVEL_TILING_H
