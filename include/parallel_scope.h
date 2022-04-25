#ifndef PARALLEL_SCOPE_H
#define PARALLEL_SCOPE_H

#include <string>
#include <variant>

#include <container_utils.h>
#include <except.h>
#include <hash_combine.h>

namespace ir {

struct SerialScope {};
inline bool operator==(const SerialScope &lhs, const SerialScope &rhs) {
    return true;
}
inline bool operator!=(const SerialScope &lhs, const SerialScope &rhs) {
    return false;
}

struct OpenMPScope {};
inline bool operator==(const OpenMPScope &lhs, const OpenMPScope &rhs) {
    return true;
}
inline bool operator!=(const OpenMPScope &lhs, const OpenMPScope &rhs) {
    return false;
}
inline std::string toString(const OpenMPScope &parallel) { return "openmp"; }

struct CUDAStreamScope {};
inline bool operator==(const CUDAStreamScope &lhs, const CUDAStreamScope &rhs) {
    return true;
}
inline bool operator!=(const CUDAStreamScope &lhs, const CUDAStreamScope &rhs) {
    return false;
}
inline std::string toString(const CUDAStreamScope &parallel) {
    return "cudastream";
}

struct CUDAScope {
    enum Level { Block, Thread } level_;
    enum Dim { X, Y, Z } dim_;
};
inline bool operator==(const CUDAScope &lhs, const CUDAScope &rhs) {
    return lhs.level_ == rhs.level_ && lhs.dim_ == rhs.dim_;
}
inline bool operator!=(const CUDAScope &lhs, const CUDAScope &rhs) {
    return !(lhs == rhs);
}
inline std::string toString(const CUDAScope &parallel) {
    std::string ret;
    switch (parallel.level_) {
    case CUDAScope::Level::Block:
        ret = "blockIdx";
        break;
    case CUDAScope::Level::Thread:
        ret = "threadIdx";
        break;
    default:
        ASSERT(false);
    }
    switch (parallel.dim_) {
    case CUDAScope::Dim::X:
        ret += ".x";
        break;
    case CUDAScope::Dim::Y:
        ret += ".y";
        break;
    case CUDAScope::Dim::Z:
        ret += ".z";
        break;
    default:
        ASSERT(false);
    }
    return ret;
}

// The first type is default
typedef std::variant<SerialScope, OpenMPScope, CUDAStreamScope, CUDAScope>
    ParallelScope;

inline std::string toString(const ParallelScope &parallel) {
    if (std::holds_alternative<SerialScope>(parallel)) {
        return "";
    } else if (std::holds_alternative<OpenMPScope>(parallel)) {
        return toString(std::get<OpenMPScope>(parallel));
    } else if (std::holds_alternative<CUDAScope>(parallel)) {
        return toString(std::get<CUDAScope>(parallel));
    } else if (std::holds_alternative<CUDAStreamScope>(parallel)) {
        return toString(std::get<CUDAStreamScope>(parallel));
    } else {
        ASSERT(false);
    }
}

inline ParallelScope parseParallelScope(const std::string &_str) {
    auto &&str = tolower(_str);
    if (auto scope = SerialScope{}; str == tolower(toString(scope))) {
        return scope;
    }
    if (auto scope = OpenMPScope{}; str == tolower(toString(scope))) {
        return scope;
    }
    if (auto scope = CUDAStreamScope{}; str == tolower(toString(scope))) {
        return scope;
    }
    for (auto &&level : {CUDAScope::Block, CUDAScope::Thread}) {
        for (auto &&dim : {CUDAScope::X, CUDAScope::Y, CUDAScope::Z}) {
            if (auto scope = CUDAScope{level, dim};
                str == tolower(toString(scope))) {
                return scope;
            }
        }
    }
    ERROR("Unrecognized parallel scope " + _str);
}

constexpr ParallelScope serialScope = SerialScope{};

constexpr ParallelScope threadIdxX = CUDAScope{CUDAScope::Thread, CUDAScope::X};
constexpr ParallelScope threadIdxY = CUDAScope{CUDAScope::Thread, CUDAScope::Y};
constexpr ParallelScope threadIdxZ = CUDAScope{CUDAScope::Thread, CUDAScope::Z};
constexpr ParallelScope blockIdxX = CUDAScope{CUDAScope::Block, CUDAScope::X};
constexpr ParallelScope blockIdxY = CUDAScope{CUDAScope::Block, CUDAScope::Y};
constexpr ParallelScope blockIdxZ = CUDAScope{CUDAScope::Block, CUDAScope::Z};

} // namespace ir

namespace std {

template <> struct hash<ir::SerialScope> {
    size_t operator()(const ir::SerialScope &) { return 0; }
};

template <> struct hash<ir::OpenMPScope> {
    size_t operator()(const ir::OpenMPScope &) { return 0; }
};

template <> struct hash<ir::CUDAStreamScope> {
    size_t operator()(const ir::CUDAStreamScope &) { return 0; }
};

template <> struct hash<ir::CUDAScope> {
    size_t operator()(const ir::CUDAScope &parallel) {
        return ir::hashCombine(std::hash<int>()((int)parallel.level_),
                               std::hash<int>()((int)parallel.dim_));
    }
};

} // namespace std

#endif // PARALLEL_SCOPE_H
