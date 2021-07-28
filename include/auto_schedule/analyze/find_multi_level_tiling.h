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

struct ForsWithDataReuse {
    std::vector<ForInfo> spaceLoops;
    std::vector<ForInfo> reductionLoops;
};

class FindMultiLevelTiling : public Visitor {
    std::vector<ForInfo> stack_;
    std::vector<ForsWithDataReuse> found_;
    bool innermost_;

  public:
    std::vector<ForsWithDataReuse> result() { return found_; }

  protected:
    void visit(const For &op);
};

} // namespace ir

#endif // IR_FIND_MULTI_LEVEL_TILING_H
