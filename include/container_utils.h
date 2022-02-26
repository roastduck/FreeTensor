#ifndef CONTAINER_UTILS_H
#define CONTAINER_UTILS_H

#include <unordered_map>
#include <unordered_set>

namespace ir {

template <class T, class V1, class V2, class Hash, class KeyEqual>
std::unordered_map<T, std::pair<V1, V2>, Hash, KeyEqual>
intersect(const std::unordered_map<T, V1, Hash, KeyEqual> &lhs,
          const std::unordered_map<T, V2, Hash, KeyEqual> &rhs) {
    std::unordered_map<T, std::pair<V1, V2>, Hash, KeyEqual> ret;
    for (auto &&[key, v1] : lhs) {
        if (rhs.count(key)) {
            ret.emplace(key, std::make_pair(v1, rhs.at(key)));
        }
    }
    return ret;
}

template <class T, class Hash, class KeyEqual>
bool hasIntersect(const std::unordered_set<T, Hash, KeyEqual> &lhs,
                  const std::unordered_set<T, Hash, KeyEqual> &rhs) {
    for (auto &&item : lhs) {
        if (rhs.count(item)) {
            return true;
        }
    }
    return false;
}

} // namespace ir

#endif // CONTAINER_UTILS_H
