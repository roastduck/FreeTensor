#include <id.h>

namespace freetensor {

std::atomic_int64_t ID::globalIdCnt_ = 0;

std::ostream &operator<<(std::ostream &os, const ID &id) {
    return os << id.id_;
}

bool operator==(const ID &lhs, const ID &rhs) { return lhs.id_ == rhs.id_; }

} // namespace freetensor

namespace std {

size_t hash<freetensor::ID>::operator()(const freetensor::ID &id) const {
    return std::hash<int64_t>()(id.id_);
}

} // namespace std
