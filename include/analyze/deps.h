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
    int defAxis_;
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
    std::unordered_map<int, std::string> axis2loop_; // axis in space -> ForNode

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
    const std::unordered_map<int, std::string> &axis2loop() const {
        return axis2loop_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        // For a[i] = a[i] + 1, write happens after read
        cur_.emplace_back(makeIntConst(0));
        auto ap = Ref<AccessPoint>::make();
        *ap = {op, defAxis_.at(op->var_), cur_, op->indices_};
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

enum class FindDepsMode : int {
    Normal,
    Inv,
    NonZero,
};

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps : public Visitor {
    const std::unordered_map<const ASTNode *, Ref<AccessPoint>> &points_;
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads_,
        &writes_;
    const std::unordered_map<const ASTNode *, LinearExpr> &linear_;

    FindDepsMode mode_;
    // callback(var name) -> [loop IDs]: Get loops that we are intereseted in
    typedef std::function<std::vector<int>(const std::string &)> LoopsCallback;
    const LoopsCallback &loops_;
    // callback(axis, var name, later access, earlier access): Called when we
    // found a interesting dependency
    typedef std::function<void(int, const std::string &, const AST &,
                               const AST &)>
        FoundCallback;
    const FoundCallback &found_;

    isl_ctx *isl_;

  public:
    AnalyzeDeps(
        const std::unordered_map<const ASTNode *, Ref<AccessPoint>> &points,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &writes,
        const std::unordered_map<const ASTNode *, LinearExpr> &linear,
        FindDepsMode mode, const LoopsCallback &loops,
        const FoundCallback &found)
        : points_(points), reads_(reads), writes_(writes), linear_(linear),
          mode_(mode), loops_(loops), found_(found) {
        isl_ = isl_ctx_alloc();
    }

    ~AnalyzeDeps() { isl_ctx_free(isl_); }

  private:
    std::string linear2str(const LinearExpr &lin) const;
    std::string makeIterList(const std::vector<Expr> &list, int eraseBefore,
                             int n) const;
    std::string makeLinList(const std::vector<LinearExpr> &list) const;
    std::string makeRange(const std::vector<Expr> &list) const;
    std::string makeNdList(const std::string &name, int n) const;
    std::string makeAccMap(const AccessPoint &p, int iterDim, int accDim) const;
    std::string makeSingleIneq(FindDepsMode mode, int iterId,
                               int iterDim) const;

    static const std::string &getVar(const AST &op);

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
 * @param loop : callback(var name) -> [ID] of the interesting loops
 * @param found : callback(loop ID, var name, later op, erlier op)
 */
void findDeps(
    const Stmt &op, FindDepsMode mode,
    const std::function<std::vector<std::string>(const std::string &)> &loops,
    const std::function<void(const std::string &, const std::string &,
                             const AST &, const AST &)> &found);

}; // namespace ir

#endif // DEPS_H
