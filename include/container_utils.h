#ifndef FREE_TENSOR_CONTAINER_UTILS_H
#define FREE_TENSOR_CONTAINER_UTILS_H

#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <except.h>

namespace freetensor {

template <class T, class V1, class V2, class Hash, class KeyEqual>
std::unordered_map<T, std::pair<V1, V2>, Hash, KeyEqual>
intersect(const std::unordered_map<T, V1, Hash, KeyEqual> &lhs,
          const std::unordered_map<T, V2, Hash, KeyEqual> &rhs) {
    std::unordered_map<T, std::pair<V1, V2>, Hash, KeyEqual> ret;
    ret.reserve(std::min(lhs.size(), rhs.size()));
    for (auto &&[key, v1] : lhs) {
        if (rhs.count(key)) {
            ret.emplace(key, std::make_pair(v1, rhs.at(key)));
        }
    }
    return ret;
}

template <class T, class Hash, class KeyEqual>
std::unordered_set<T, Hash, KeyEqual>
intersect(const std::unordered_set<T, Hash, KeyEqual> &lhs,
          const std::unordered_set<T, Hash, KeyEqual> &rhs) {
    std::unordered_set<T, Hash, KeyEqual> ret;
    ret.reserve(std::min(lhs.size(), rhs.size()));
    for (auto &&key : lhs) {
        if (rhs.count(key)) {
            ret.emplace(key);
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

template <class T>
std::vector<T> intersect(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    std::vector<T> ret;
    ret.reserve(std::min(lhs.size(), rhs.size()));
    for (auto &&item : lhs) {
        if (std::find(rhs.begin(), rhs.end(), item) != rhs.end()) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

template <class T, class Hash, class KeyEqual>
bool isSubSetOf(const std::unordered_set<T, Hash, KeyEqual> &lhs,
                const std::unordered_set<T, Hash, KeyEqual> &rhs) {
    for (auto &&item : lhs) {
        if (!rhs.count(item)) {
            return false;
        }
    }
    return true;
}

template <class T>
std::vector<T> uni(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    std::vector<T> ret;
    ret.reserve(std::max(lhs.size(), rhs.size()));
    ret.insert(ret.end(), lhs.begin(), lhs.end());
    for (auto &&item : rhs) {
        if (std::find(lhs.begin(), lhs.end(), item) == lhs.end()) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

template <class T>
std::vector<T> cat(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    std::vector<T> ret;
    ret.reserve(lhs.size() + rhs.size());
    ret.insert(ret.end(), lhs.begin(), lhs.end());
    ret.insert(ret.end(), rhs.begin(), rhs.end());
    return ret;
}

template <class T, class U>
std::vector<T> filter(const std::vector<T> &vec, const U &callback) {
    std::vector<T> ret;
    ret.reserve(vec.size());
    for (auto item : vec) {
        if (callback(item)) {
            ret.emplace_back(item);
        }
    }
    return ret;
}

inline std::string tolower(const std::string &s) {
    std::string ret;
    ret.reserve(s.length());
    for (char c : s) {
        ret.push_back(std::tolower(c));
    }
    return ret;
}

/**
 * Python-like slicing that supports negative indexing as reversed indexing
 */
inline std::string slice(const std::string &s, int _begin, int _end) {
    int begin = _begin >= 0 ? _begin : s.length() + _begin;
    int end = _end >= 0 ? _end : s.length() + _end;
    int len = end - begin;
    ASSERT(len >= 0);
    return s.substr(begin, len);
}
inline std::string slice(const std::string &s, int begin) {
    return slice(s, begin, s.length());
}

} // namespace freetensor

#endif // FREE_TENSOR_CONTAINER_UTILS_H
