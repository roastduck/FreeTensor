#ifndef IR_AUTO_SCHEDULE_STRUCT_H
#define IR_AUTO_SCHEDULE_STRUCT_H
#include <array>
#include <ast.h>
#include <hash.h>
#include <vector>

namespace ir {
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
    std::vector<SubTree<ExprNode>> indices;
    std::vector<std::vector<SubTree<ExprNode>>> checkDataReuseIndices;
};

struct ElementWiseInfo {
    Stmt stmt;
    std::vector<ForInfo> fors;
    [[nodiscard]] bool isValid() const { return !fors.empty(); }
};

} // namespace ir

template <> struct std::hash<ir::ForInfo> {
    std::size_t operator()(ir::ForInfo const &s) const noexcept {
        std::size_t h = std::hash<ir::ID>{}(s.id);
        h = ir::hashCombine(h, std::hash<std::int64_t>{}(s.length));
        return h;
    }
};

template <> struct std::hash<ir::ElementWiseInfo> {
    std::size_t operator()(ir::ElementWiseInfo const &s) const noexcept {
        std::size_t h = std::hash<ir::Stmt>{}(s.stmt.get());
        for (const auto &f : s.fors) {
            h = ir::hashCombine(h, std::hash<ir::ForInfo>{}(f));
        }
        return h;
    }
};

template <> struct std::hash<ir::ForsWithDataReuse> {
    std::size_t operator()(ir::ForsWithDataReuse const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &f : s.spaceLoops)
            h = ir::hashCombine(h, std::hash<ir::ForInfo>{}(f));
        for (const auto &f : s.reductionLoops)
            h = ir::hashCombine(h, std::hash<ir::ForInfo>{}(f));
        return h;
    }
};

template <> struct std::hash<ir::MultiLevelTilingAnnotation> {
    std::size_t
    operator()(ir::MultiLevelTilingAnnotation const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &t : s.spaceLoopTiling)
            for (const auto i : t)
                h = ir::hashCombine(h, std::hash<int>{}(i));
        for (const auto &t : s.reductionLoopTiling)
            for (const auto i : t)
                h = ir::hashCombine(h, std::hash<int>{}(i));
        return h;
    }
};
#endif