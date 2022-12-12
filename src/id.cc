#include <id.h>

namespace freetensor {

std::atomic_uint64_t ID::globalIdCnt_ = 1;

int OSTREAM_NO_ID_SIGN = std::ostream::xalloc();
std::function<std::ostream &(std::ostream &)> manipNoIdSign(bool flag) {
    return [flag](std::ostream &os) -> std::ostream & {
        os.iword(OSTREAM_NO_ID_SIGN) = flag;
        return os;
    };
}

std::ostream &operator<<(std::ostream &os, const ID &id) {
    if (!os.iword(OSTREAM_NO_ID_SIGN)) {
        os << '#';
    }
    return os << id.id_;
}

bool operator==(const ID &lhs, const ID &rhs) { return lhs.id_ == rhs.id_; }

} // namespace freetensor

namespace std {

size_t hash<freetensor::ID>::operator()(const freetensor::ID &id) const {
    return std::hash<uint64_t>()(id.id_);
}

} // namespace std
