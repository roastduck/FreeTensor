#ifndef IR_FIND_MULTI_LEVEL_TILING_H
#define IR_FIND_MULTI_LEVEL_TILING_H

#include <analyze/find_loop_variance.h>
#include <ast.h>
#include <auto_schedule/structs.h>
#include <hash.h>
#include <pass/undo_make_reduction.h>
#include <unordered_map>
#include <vector>
#include <visitor.h>

namespace ir {

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
    std::vector<Expr> bufIndices_;
    std::vector<std::vector<Expr>> bufCheckDataReuseIndices_;
    std::string dest_;
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

inline std::vector<ID> fakeFindMultiLevelTiling(const Stmt &ast) {
    std::vector<ForsWithDataReuse> src = findMultiLevelTiling(ast);
    std::vector<ID> ret;
    std::string s("S "), r("R "), sp(" ");
    for (unsigned i = 0; i < src.size(); i++) {
        std::string item;
        const ForsWithDataReuse &nw = src[i];
        for (const auto &loop : nw.spaceLoops) {
            item.append(s);
            item.append(loop.id.strId());
            item.append(sp);
        }
        for (const auto &loop : nw.reductionLoops) {
            item.append(r);
            item.append(loop.id.strId());
            item.append(sp);
        }
        ret.push_back(item);
    }
    return ret;
}

} // namespace ir

#endif // IR_FIND_MULTI_LEVEL_TILING_H
