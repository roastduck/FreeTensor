#ifndef FREE_TENSOR_CONTAINER_UTILS_H
#define FREE_TENSOR_CONTAINER_UTILS_H

#include <algorithm>
#include <cctype>
#include <iostream>
#include <ranges>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <range/v3/range.hpp>
#include <range/v3/view.hpp>

namespace freetensor {

namespace views = ranges::views;

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
template <class T, class Hash, class KeyEqual>
std::unordered_set<T, Hash, KeyEqual>
uni(const std::unordered_set<T, Hash, KeyEqual> &lhs,
    const std::unordered_set<T, Hash, KeyEqual> &rhs) {
    if (lhs.size() < rhs.size()) {
        return uni(rhs, lhs);
    }
    auto ret = lhs;
    for (auto &&item : rhs) {
        ret.insert(item);
    }
    return ret;
}

template <class T, class Hash, class KeyEqual>
std::unordered_set<T, Hash, KeyEqual>
diff(const std::unordered_set<T, Hash, KeyEqual> &lhs,
     const std::unordered_set<T, Hash, KeyEqual> &rhs) {
    auto ret = lhs;
    for (auto &&item : rhs) {
        ret.erase(item);
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
std::string slice(const std::string &s, int begin, int end);
inline std::string slice(const std::string &s, int begin) {
    return slice(s, begin, s.length());
}

// clang-format off
/**
 * Comma-joined print of any range
 */
template <class T>
requires std::ranges::range<T> && (!std::convertible_to<T, std::string>)
std::ostream &operator<<(std::ostream &os, const T &r) {
    for (auto &&[i, item] : views::enumerate(r)) {
        os << (i > 0 ? ", " : "") << item;
    }
    return os;
}
// clang-format on

/**
 * Comma-joined print of any tuple
 */
template <typename... Ts>
std::ostream &operator<<(std::ostream &os, std::tuple<Ts...> const &tuple) {
    return std::apply(
        [&os](Ts const &...t) -> std::ostream & {
            int i = 0;
            return ((os << (i++ > 0 ? ", " : "") << t), ...);
        },
        tuple);
}

/**
 * Join a sequence of elements to a string with given splitter
 */
struct _Join {
    const std::string &splitter;
};

template <std::ranges::range Container>
std::string join(Container &&c, const std::string &splitter) {
    std::ostringstream oss;
    bool first = true;
    for (const auto &s : c) {
        if (!first)
            oss << splitter;
        oss << s;
        first = false;
    }
    return oss.str();
}

inline auto join(const std::string &splitter) { return _Join{splitter}; }

template <std::ranges::range Container>
auto operator|(Container &&c, const _Join &joiner) {
    return join(std::forward<Container>(c), joiner.splitter);
}

} // namespace freetensor

#endif // FREE_TENSOR_CONTAINER_UTILS_H
