#ifndef IR_FIND_THREAD_BIND_H
#define IR_FIND_THREAD_BIND_H

#include <analyze/find_loop_variance.h>
#include <tuple>
#include <vector>
#include <visitor.h>

namespace ir {
class FindThreadBind : public Visitor {
    std::vector<std::pair<std::string, int>> stackId_;
    std::vector<bool> stackMarkBranch_; // mark whether have multiple children
    std::vector<std::pair<std::string, int>> bufId_;

    std::vector<std::pair<std::string, int>> found_;

    bool downward_ = true;

  public:
    FindThreadBind() {}
    std::vector<std::pair<std::string, int>> result();

  protected:
    void visit(const For &op) override;
};

inline std::vector<std::pair<std::string, int>> findThreadBind(const AST &ast) {
    FindThreadBind find;
    find(ast);
    return find.result();
}
} // namespace ir

#endif // IR_FIND_THREAD_BIND_H
