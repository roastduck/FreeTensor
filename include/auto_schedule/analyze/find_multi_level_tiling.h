#ifndef IR_FIND_MULTI_LEVEL_TILING_H
#define IR_FIND_MULTI_LEVEL_TILING_H

#include <analyze/find_loop_variance.h>
#include <unordered_map>
#include <vector>
#include <visitor.h>

namespace ir {

struct ForInfo {
    std::string id;
    int length;
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

} // namespace ir

#endif // IR_FIND_MULTI_LEVEL_TILING_H
