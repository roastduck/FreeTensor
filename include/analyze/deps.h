#ifndef DEPS_H
#define DEPS_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>

#include <analyze/linear.h>
#include <visitor.h>

namespace ir {

struct AccessPoint {
    AST op_;
    std::vector<Expr> iter_, access_;
};

/**
 * Find read and write points
 */
class FindAccessPoint : public Visitor {
    std::vector<Expr> cur_; // Current iteration point in the space
    std::unordered_map<const ASTNode *, Ref<AccessPoint>> points_;
    std::unordered_multimap<std::string, Ref<AccessPoint>> reads_, writes_;
    std::unordered_map<std::string, int> loop2axis_; // ForNode -> axis in space

    // Var name -> axis: Which axis is a local var defined
    std::unordered_map<std::string, int> defAxis_;

  public:
    const std::unordered_map<const ASTNode *, Ref<AccessPoint>> &
    points() const {
        return points_;
    }
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &
    reads() const {
        return reads_;
    }
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &
    writes() const {
        return writes_;
    }
    const std::unordered_map<std::string, int> &loop2axis() const {
        return loop2axis_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        // For a[i] = a[i] + 1, write happens after read
        cur_.emplace_back(makeIntConst(0));
        auto ap = Ref<AccessPoint>::make();
        *ap = {op, cur_, op->indices_};
        std::fill(ap->iter_.begin(), ap->iter_.begin() + defAxis_.at(op->var_),
                  makeIntConst(0));
        points_.emplace(op.get(), ap);
        writes_.emplace(op->var_, ap);

        cur_.back() = makeIntConst(1);
        Visitor::visit(op);
        cur_.pop_back();
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const AddTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps : public Visitor {
    const std::unordered_map<const ASTNode *, Ref<AccessPoint>> &points_;
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads_,
        &writes_;

    // Permuting loops that we are intereseted in
    const std::vector<int> &loops_;

    const std::unordered_map<const ASTNode *, LinearExpr> &linear_;

    const std::function<void(int, const AST &, const AST &)> &callback_;

    isl_ctx *isl_;

  public:
    AnalyzeDeps(
        const std::unordered_map<const ASTNode *, Ref<AccessPoint>> &points,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &writes,
        const std::vector<int> &loops,
        const std::unordered_map<const ASTNode *, LinearExpr> &linear,
        const std::function<void(int, const AST &, const AST &)> &callback)
        : points_(points), reads_(reads), writes_(writes), loops_(loops),
          linear_(linear), callback_(callback) {
        isl_ = isl_ctx_alloc();
    }

    ~AnalyzeDeps() { isl_ctx_free(isl_); }

  private:
    std::string linear2str(const LinearExpr &lin) const;
    std::string makeIterList(const std::vector<Expr> &list, int n) const;
    std::string makeLinList(const std::vector<LinearExpr> &list) const;
    std::string makeRange(const std::vector<Expr> &list) const;
    std::string makeNdList(const std::string &name, int n) const;
    std::string makeAccMap(const AccessPoint &p, int iterDim, int accDim) const;
    std::string makeSingleIneq(int iterId, int iterDim) const;

    void checkDep(const AccessPoint &lhs, const AccessPoint &rhs);

    template <class T> void visitStoreLike(const T &op) {
        Visitor::visit(op);
        auto &&point = points_.at(op.get());
        auto range = reads_.equal_range(op->var_);
        for (auto i = range.first; i != range.second; i++) {
            checkDep(*point, *(i->second)); // WAR
        }
        range = writes_.equal_range(op->var_);
        for (auto i = range.first; i != range.second; i++) {
            if (op->nodeType() != ASTNodeType::Store &&
                i->second->op_->nodeType() != ASTNodeType::Store) {
                // No dependency between reductions
                continue;
            }
            checkDep(*point, *(i->second)); // WAW
        }
    }

  protected:
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const AddTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

/**
 * Find all inverse (negative) dependencies along the given loops
 *
 * @param op : AST root
 * @param loop : ID of the interesting loops
 * @param callback : f(loop ID, later op, erlier op)
 */
void findInvDeps(const Stmt &op, const std::vector<std::string> loops,
                 const std::function<void(const std::string &, const AST &,
                                          const AST &)> &callback);

}; // namespace ir

#endif // DEPS_H
