#include <analyze/all_defs.h>
#include <schedule.h>

namespace freetensor {

void Schedule::autoSetMemType(const Target &target) {
    // Try to put each VarDef as near to processor as possible
    if (target.type() == TargetType::GPU) {
        for (auto &&[defId, name] : allDefs(ast(), {AccessType::Cache})) {
            try {
                setMemType(defId, MemType::GPULocal);
            } catch (const InvalidSchedule &e) {
                try {
                    setMemType(defId, MemType::GPUShared);
                } catch (const InvalidSchedule &e) {
                    // do nothing
                }
            }
        }
    }
}

} // namespace freetensor
