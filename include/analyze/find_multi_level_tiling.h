#ifndef IR_FIND_MULTI_LEVEL_TILING_H
#define IR_FIND_MULTI_LEVEL_TILING_H

#include <analyze/find_loop_variance.h>
#include <unordered_map>
#include <vector>
#include <visitor.h>
#include <boost/functional/hash.hpp>

namespace ir {

struct ForInfo {
    std::string id;
    int64_t length;
};

struct ForsWithDataReuse {
    std::vector<ForInfo> spaceLoops;
    std::vector<ForInfo> reductionLoops;
};

struct ForWithStore {
    std::string id;
    std::vector<SubTree<ExprNode>> indices;
    std::vector<std::vector<SubTree<ExprNode>>> checkDataReuseIndices;
};

class FindHasStore : public Visitor {
    std::vector<ForInfo> stack_;
    std::unordered_map<std::string, ForWithStore> found_;

  public:
    std::unordered_map<std::string, ForWithStore> result() { return found_; }

  protected:
    void visit(const For &op) override;
    void visit(const Store &op) override;
    void visit(const Load &op) override;
};

class FindMultiLevelTiling : public Visitor {
    // storeBuf() must be called after calling this visitor

    std::vector<ForInfo> stack_;
    std::vector<bool> stackMarkBranch_; // mark whether have multiple children

    std::vector<ForInfo> buf_;
    std::vector<SubTree<ExprNode>> bufIndices_;
    std::vector<std::vector<SubTree<ExprNode>>> bufCheckDataReuseIndices_;
    bool downward = true;

    std::vector<ForsWithDataReuse> found_;

    std::unordered_map<std::string, ForWithStore> &forsWithStore_;
    LoopVariExprMap &loopVariExprMap_;

  public:
    FindMultiLevelTiling(
        std::unordered_map<std::string, ForWithStore> &forsWithStore,
        LoopVariExprMap &loopVariExprMap)
        : forsWithStore_(forsWithStore), loopVariExprMap_(loopVariExprMap) {}
    std::vector<ForsWithDataReuse> result() { return found_; }
    void storeBuf();

  protected:
    void visit(const For &op) override;

  private:
    bool hasStore(const For &op);
};

inline std::vector<ForsWithDataReuse> findMultiLevelTiling(const Stmt &ast) {
    FindHasStore findHasStore;
    findHasStore(ast);
    auto forsWithStore = findHasStore.result();
    auto loopVariExprMap = findLoopVariance(ast).first;

    FindMultiLevelTiling find(forsWithStore, loopVariExprMap);
    find(ast);
    find.storeBuf();
    return find.result();
}

inline std::vector<std::string> fakeFindMultiLevelTiling(const Stmt &ast) {
    std::vector<ForsWithDataReuse> src = findMultiLevelTiling(ast);
    std::vector<std::string> ret;
    std::string s("S "), r("R "), sp(" ");
    for (unsigned i = 0; i < src.size(); i++) {
        std::string item;
        const ForsWithDataReuse &nw = src[i];
        for (const auto &loop : nw.spaceLoops) {
            item.append(s);
            item.append(loop.id);
            item.append(sp);
        }
        for (const auto &loop : nw.reductionLoops) {
            item.append(r);
            item.append(loop.id);
            item.append(sp);
        }
        ret.push_back(item);
    }
    return ret;
}

} // namespace ir

template <> struct std::hash<ir::ForInfo> {
    std::size_t operator()(ir::ForInfo const &s) const noexcept {
        std::size_t h = std::hash<std::string>{}(s.id);
        boost::hash_combine(h, std::hash<std::int64_t>{}(s.length));
        return h;
    }
};

template <> struct std::hash<ir::ForsWithDataReuse> {
    std::size_t operator()(ir::ForsWithDataReuse const &s) const noexcept {
        std::size_t h = 0;
        for (const auto &f : s.spaceLoops)
            boost::hash_combine(h, std::hash<ir::ForInfo>{}(f));
        for (const auto &f : s.reductionLoops)
            boost::hash_combine(h, std::hash<ir::ForInfo>{}(f));
        return h;
    }
};
#endif // IR_FIND_MULTI_LEVEL_TILING_H
