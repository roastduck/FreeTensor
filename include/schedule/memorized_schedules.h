#ifndef FREE_TENSOR_MEMORIZED_SCHEDULES_H
#define FREE_TENSOR_MEMORIZED_SCHEDULES_H

#include <mutex>
#include <unordered_set>

#include <schedule/schedule_log.h>

namespace freetensor {

/**
 * Storage of all tried scheduling decisions for all `Schedule`s `fork`ed from a
 * common one
 *
 * When we are searching for best scheduling decisions, we often try similar
 * ones, for example `Original -> Schedule A -> Schedule B` and `Original ->
 * Schedule A -> Schedule C`. Looking up from this class saves time for applying
 * identical decisions
 *
 * This class is thread-safe
 *
 * This class is not named cache or storage, to avoid confusion with hardware
 * features
 */
class MemorizedSchedules {
    std::unordered_set<ScheduleLog> memorized_;
    std::mutex lock_;

  public:
    /**
     * Lookup for a particular schedule
     *
     * If there is a memorized result, return the memorized one to save memory
     * (so the shared linked lists form a tree). If not found, return the new
     * log
     */
    ScheduleLog lookup(const ScheduleLog &log) {
        std::lock_guard<std::mutex> guard(lock_);
        if (auto it = memorized_.find(log); it != memorized_.end()) {
            return *it;
        } else {
            return log;
        }
    }

    /**
     * Save a new log
     */
    void save(const ScheduleLog &log) {
        std::lock_guard<std::mutex> guard(lock_);
        memorized_.insert(log);
    }
};

} // namespace freetensor

#endif // FREE_TENSOR_MEMORIZED_SCHEDULES_H
