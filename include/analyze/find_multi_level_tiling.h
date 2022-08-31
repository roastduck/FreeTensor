#ifndef FREE_TENSOR_FIND_MULTI_LEVEL_TILING_H
#define FREE_TENSOR_FIND_MULTI_LEVEL_TILING_H

#include <analyze/find_loop_variance.h>
#include <ast.h>
#include <auto_schedule/structs.h>
#include <hash.h>
#include <pass/undo_make_reduction.h>
#include <unordered_map>
#include <vector>
#include <visitor.h>

namespace freetensor {

class FindHasStore : public Visitor {
    std::vector<ForInfo> stack_;
    std::unordered_map<ID, ForWithStore> found_;

  public:
    std::unordered_map<ID, ForWithStore> result() { return found_; }

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
    ForWithStore nowFor_;
    Store nowInit_;
    bool downward = true;

    std::vector<ForsWithDataReuse> found_;

    std::unordered_map<ID, ForWithStore> &forsWithStore_;
    LoopVariExprMap &loopVariExprMap_;

  public:
    FindMultiLevelTiling(std::unordered_map<ID, ForWithStore> &forsWithStore,
                         LoopVariExprMap &loopVariExprMap)
        : forsWithStore_(forsWithStore), loopVariExprMap_(loopVariExprMap) {}
    std::vector<ForsWithDataReuse> result() { return found_; }
    void storeBuf();

  protected:
    void visit(const For &op) override;

  private:
    std::string hasStore(const For &op);
};

inline std::vector<ForsWithDataReuse> findMultiLevelTiling(const Stmt &ast) {
    auto ast_wo_reduction = undoMakeReduction(ast);
    FindHasStore findHasStore;
    findHasStore(ast_wo_reduction);
    auto forsWithStore = findHasStore.result();
    auto loopVariExprMap = findLoopVariance(ast_wo_reduction).first;
    FindMultiLevelTiling find(forsWithStore, loopVariExprMap);
    find(ast_wo_reduction);
    find.storeBuf();
    return find.result();
}

inline std::vector<std::string> fakeFindMultiLevelTiling(const Stmt &ast) {
    std::vector<ForsWithDataReuse> src = findMultiLevelTiling(ast);
    std::vector<std::string> ret;
    std::string s("S "), r("R "), sp(" ");
    std::ostringstream oss;
    oss << manipMetadataSkipLocation;
    for (unsigned i = 0; i < src.size(); i++) {
        const ForsWithDataReuse &nw = src[i];
        for (const auto &loop : nw.spaceLoops)
            oss << "S " << loop.metadata << " ";
        for (const auto &loop : nw.reductionLoops)
            oss << "R " << loop.metadata << " ";
        ret.push_back(oss.str());
    }
    return ret;
}

} // namespace freetensor

#endif // FREE_TENSOR_FIND_MULTI_LEVEL_TILING_H
