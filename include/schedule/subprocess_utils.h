#ifndef FREE_TENSOR_SCHEDULE_SUBPROCESS_UTILS_H
#define FREE_TENSOR_SCHEDULE_SUBPROCESS_UTILS_H

#include <string>

#include <nlohmann/json.hpp>

#include <schedule.h>
#include <subprocess.h>

namespace freetensor {

using json = nlohmann::json;

inline ID idFromJson(const json &j) {
    return j.is_null() ? ID() : ID::make(j.get<uint64_t>());
}

inline Schedule::IDMap idMapFromJson(const json &j) {
    Schedule::IDMap ret;
    for (auto it = j.begin(); it != j.end(); ++it) {
        ret[ID::make(std::stoull(it.key()))] = idFromJson(it.value());
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_SCHEDULE_SUBPROCESS_UTILS_H
