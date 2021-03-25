#ifndef SINK_VAR_H
#define SINK_VAR_H

#include <set>
#include <unordered_set>
#include <vector>

#include <mutator.h>
#include <visitor.h>

namespace ir {

class FindAllLoops : public Visitor {
    std::vector<std::string> loops_;

  public:
    const std::vector<std::string> loops() const { return loops_; }

  protected:
    void visit(const For &op) override;
};

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

} // namespace ir

#endif // SINK_VAR_H
