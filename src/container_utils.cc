#include <container_utils.h>
#include <except.h>

namespace freetensor {

std::string slice(const std::string &s, int _begin, int _end) {
    int begin = _begin >= 0 ? _begin : s.length() + _begin;
    int end = _end >= 0 ? _end : s.length() + _end;
    int len = end - begin;
    ASSERT(len >= 0);
    return s.substr(begin, len);
}

} // namespace freetensor
