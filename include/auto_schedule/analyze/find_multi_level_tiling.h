#ifndef IR_FIND_MULTI_LEVEL_TILING_H
#define IR_FIND_MULTI_LEVEL_TILING_H

#include <vector>
#include <visitor.h>

namespace ir {

struct ForInfo {
    std::string id;
    int begin;
    int end;
};

struct ThreeNestedFors {
    ForInfo i, j, k;
};

class FindMultiLevelTiling : public Visitor {
    std::vector<ForInfo> stack_;
    std::vector<ThreeNestedFors> found_;
    bool innermost_;

  public:
    std::vector<ThreeNestedFors> result() { return found_; }

  protected:
    void visit(const For &op);
};

} // namespace ir

#endif // IR_FIND_MULTI_LEVEL_TILING_H
