#ifndef FREE_TENSOR_AUTO_SCHEDULE_STRUCT_H
#define FREE_TENSOR_AUTO_SCHEDULE_STRUCT_H
#include <array>
#include <ast.h>
#include <hash.h>
#include <vector>

namespace freetensor {
struct MultiLevelTilingAnnotation {
    std::vector<std::vector<int>> spaceLoopTiling;
    std::vector<std::vector<int>> reductionLoopTiling;
};

struct ForInfo {
    ID id;
    int index{-1};
    int64_t length{};
    bool operator<(const ForInfo &f) const { return index < f.index; }
};

struct ForsWithDataReuse {
    std::vector<ForInfo> spaceLoops;
    std::vector<ForInfo> reductionLoops;
    std::vector<bool> dimIterated;
    std::string dest;
    ID outermost;
};

struct ForWithStore {
    ID id;
    std::string dest;
    std::vector<Expr> indices;
    std::vector<std::vector<Expr>> checkDataReuseIndices;
};

struct ElementWiseInfo {
    std::vector<ForInfo> fors;
    [[nodiscard]] bool isValid() const { return !fors.empty(); }
};

} // namespace freetensor

template <> struct std::hash<freetensor::ForInfo> {
    std::size_t operator()(freetensor::ForInfo const &s) const noexcept {
        std::size_t h = std::hash<freetensor::ID>{}(s.id);
        h = freetensor::hashCombine(h, std::hash<std::int64_t>{}(s.length));
        return h;
    }
};

template <> struct std::hash<freetensor::ElementWiseInfo> {
    std::size_t
    operator()(freetensor::ElementWiseInfo const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &f : s.fors) {
            h = freetensor::hashCombine(h, std::hash<freetensor::ForInfo>{}(f));
        }
        return h;
    }
};

template <> struct std::hash<freetensor::ForsWithDataReuse> {
    std::size_t
    operator()(freetensor::ForsWithDataReuse const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &f : s.spaceLoops)
            h = freetensor::hashCombine(h, std::hash<freetensor::ForInfo>{}(f));
        for (const auto &f : s.reductionLoops)
            h = freetensor::hashCombine(h, std::hash<freetensor::ForInfo>{}(f));
        return h;
    }
};

template <> struct std::hash<freetensor::MultiLevelTilingAnnotation> {
    std::size_t
    operator()(freetensor::MultiLevelTilingAnnotation const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &t : s.spaceLoopTiling)
            for (const auto i : t)
                h = freetensor::hashCombine(h, std::hash<int>{}(i));
        for (const auto &t : s.reductionLoopTiling)
            for (const auto i : t)
                h = freetensor::hashCombine(h, std::hash<int>{}(i));
        return h;
    }
};
#endif
