#ifndef SINK_VAR_H
#define SINK_VAR_H

#include <set>
#include <unordered_set>
#include <vector>

#include <func.h>
#include <mutator.h>
#include <visitor.h>

namespace ir {

/**
 * Make the scope of a local variable smaller
 *
 * If you don't want a variable to be sinked, please set VarDefNode::pinned_
 */
class SinkVar : public Mutator {
    const std::set<std::pair<std::string, std::string>> &deps_; // {(var, loop)}
    std::unordered_set<std::string> used_;
    bool isFixPoint_ = true;

  public:
    SinkVar(const std::set<std::pair<std::string, std::string>> &deps)
        : deps_(deps) {}

    bool isFixPoint() const { return isFixPoint_; }

  protected:
    Expr visit(const Load &op) override;
    Stmt visit(const Store &op) override;
    Stmt visit(const ReduceTo &op) override;
    Stmt visit(const VarDef &op) override;
};

Stmt sinkVar(const Stmt &op);

inline Func sinkVar(const Func &func) {
    return makeFunc(func->name_, func->params_, sinkVar(func->body_),
                    func->src_);
}

} // namespace ir

#endif // SINK_VAR_H
