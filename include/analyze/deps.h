#ifndef DEPS_H
#define DEPS_H

#include <functional>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <analyze/hash.h>
#include <cursor.h>
#include <math/gen_isl_expr.h>
#include <math/isl.h>
#include <visitor.h>

namespace ir {

struct IterAxis {
    Expr iter_;
    std::string parallel_;

    IterAxis(Expr iter, const std::string &parallel = "")
        : iter_(iter), parallel_(parallel) {}
};

struct AccessPoint {
    AST op_;
    Cursor cursor_;
    VarDef def_;
    Ref<Buffer> buffer_;
    int defAxis_;                /// The position of the VarDef
    std::vector<IterAxis> iter_; /// The temporal location of the access
    std::vector<Expr> access_;   /// The spacial location of the access
    std::vector<Expr> conds_;    /// The condition (predicate) of the access
};

class FindAllNoDeps : public Visitor {
    std::unordered_map<std::string, std::vector<std::string>>
        results_; // Var name -> [loop ID]
    // FIXME: Currently, we record a var name to loop ID relation, which is not
    // rigorous because there will be different vars with the same name.
    // Recording a VarDef ID to loop ID relation may be a better choice.
    // However, VarDef may be INSIDE a loop after pass/gpu/normalize_threads,
    // and we are unable to find the VarDef ID

  public:
    const std::unordered_map<std::string, std::vector<std::string>> &
    results() const {
        return results_;
    }

  protected:
    void visit(const For &op) override;
};

class CountBandNodeWidth : public Visitor {
    int width_ = 0;
    bool lastIsLoad_ = false;

  public:
    int width() const { return width_; }

  protected:
    void visit(const Load &op) override;
    void visit(const For &op) override;
    void visit(const Store &op) override;
    void visit(const ReduceTo &op) override;
};

inline int countBandNodeWidth(const Stmt &op) {
    CountBandNodeWidth visitor;
    visitor(op);
    return visitor.width();
}

/**
 * Find read and write points
 */
class FindAccessPoint : public VisitorWithCursor {
    bool lastIsLoad_ = false;
    std::vector<IterAxis> cur_; // Current iteration point in the space
    std::vector<Expr> conds_;
    std::unordered_map<AST, Ref<AccessPoint>> points_;
    std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> reads_,
        writes_;

    // For or StmtSeq -> coordinate in space
    std::unordered_map<std::string, std::vector<IterAxis>> scope2coord_;

    // Var name -> axis: Which axis is a local var defined
    std::unordered_map<std::string, int> defAxis_;

    // Var name -> VarDef
    std::unordered_map<std::string, VarDef> defs_;

  public:
    FindAccessPoint(const Stmt &root);

    const std::unordered_map<AST, Ref<AccessPoint>> &points() const {
        return points_;
    }
    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> &
    reads() const {
        return reads_;
    }
    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> &
    writes() const {
        return writes_;
    }
    const std::unordered_map<std::string, std::vector<IterAxis>> &
    scope2coord() const {
        return scope2coord_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        Visitor::visit(op);

        if (!cur_.empty() &&
            cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
            // top is band node
            cur_.back().iter_ =
                makeIntConst(cur_.back().iter_.as<IntConstNode>()->val_ + 1);
        }
        lastIsLoad_ = false;

        auto ap = Ref<AccessPoint>::make();
        *ap = {op,
               cursor(),
               defs_.at(op->var_),
               defs_.at(op->var_)->buffer_,
               defAxis_.at(op->var_),
               cur_,
               std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
               conds_};
        points_.emplace(op, ap);
        writes_[defs_.at(op->var_)->id()].emplace_back(ap);
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
    void visit(const MatMul &op) override { (*this)(op->equivalent_); }
};

// hash -> (expr, isl name)
typedef std::unordered_map<uint64_t, std::pair<Expr, std::string>> ExternalMap;

/**
 * GenISLExpr specialized for handling external variables
 */
class GenISLExprDeps : public GenISLExpr {
    std::unordered_map<Expr, ExternalMap> externals_;
    GetHash getHash_;
    Expr parent_ = nullptr;

  public:
    const ExternalMap &externals(const Expr &op) { return externals_[op]; }

  protected:
    using GenISLExpr::visit;
    void visitExpr(const Expr &op,
                   const std::function<void(const Expr &)> &visitNode) override;
    void visit(const Load &op) override;
};

enum class DepDirection : int {
    Normal,
    Inv,
    Same,
    Different,
};

struct NodeIDOrParallelScope {
    std::string name_;
    bool isNode_;

    NodeIDOrParallelScope(const std::string &name, bool isNode = true)
        : name_(name), isNode_(isNode) {}
};

typedef std::vector<std::pair<NodeIDOrParallelScope, DepDirection>>
    FindDepsCond;

struct Dependency {
    const FindDepsCond &cond_; /// sub-condition that fails
    const std::string &var_;
    const AccessPoint &later_, &earlier_;

    // Helper functions
    const AST &later() const { return later_.op_; }
    const AST &earlier() const { return earlier_.op_; }
    const VarDef &def() const { return earlier_.def_; }
    const std::string &defId() const { return earlier_.def_->id(); }
};
typedef std::function<void(const Dependency &)> FindDepsCallback;

typedef int DepType;
const DepType DEP_WAW = 0x1;
const DepType DEP_WAR = 0x2;
const DepType DEP_RAW = 0x4;
const DepType DEP_ALL = DEP_WAW | DEP_WAR | DEP_RAW;

enum class RelaxMode : int { Possible, Necessary };
enum class FindDepsMode : int {
    Dep,         // Dependency may happen between `earlier` and `later`
    KillEarlier, // At any point in the space of `earlier`, it is dependent by
                 // `later`
    KillLater,   // At any point in the space of `later`, it is dependent on
                 // `earlier`
    KillBoth,    // KillEarlier + KillLater
};

typedef std::function<bool(const AccessPoint &later,
                           const AccessPoint &earlier)>
    FindDepsFilter;

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps : public Visitor {
    const std::unordered_map<AST, Ref<AccessPoint>> &points_;
    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
        &reads_, &writes_;
    const std::unordered_map<std::string, std::vector<IterAxis>> &scope2coord_;
    const std::unordered_map<std::string, std::vector<std::string>>
        &noDepsLists_; // Var name -> [loop ID]
    const LoopVariExprMap &variantExpr_;

    const std::vector<FindDepsCond> &cond_;
    const FindDepsCallback &found_;
    const FindDepsFilter &filter_;

    const FindDepsMode mode_;
    const DepType depType_;
    const bool ignoreReductionWAW_;
    const bool eraseOutsideVarDef_;

    std::unordered_map<std::string, std::string>
        defId_; // var name -> VarDef ID

    std::vector<std::function<void()>> tasks_;
    std::mutex lock_;

  public:
    AnalyzeDeps(
        const std::unordered_map<AST, Ref<AccessPoint>> &points,
        const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
            &reads,
        const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
            &writes,
        const std::unordered_map<std::string, std::vector<IterAxis>>
            &scope2coord,
        const std::unordered_map<std::string, std::vector<std::string>>
            &noDepsLists,
        const LoopVariExprMap &variantExpr,
        const std::vector<FindDepsCond> &cond, const FindDepsCallback &found,
        FindDepsMode mode, DepType depType, const FindDepsFilter &filter,
        bool ignoreReductionWAW, bool eraseOutsideVarDef)
        : points_(points), reads_(reads), writes_(writes),
          scope2coord_(scope2coord), noDepsLists_(noDepsLists),
          variantExpr_(variantExpr), cond_(cond), found_(found),
          filter_(filter), mode_(mode), depType_(depType),
          ignoreReductionWAW_(ignoreReductionWAW),
          eraseOutsideVarDef_(eraseOutsideVarDef) {}

    const std::vector<std::function<void()>> &tasks() const { return tasks_; }

  private:
    std::string makeIterList(GenISLExprDeps &genISLExpr,
                             const std::vector<IterAxis> &list, int n);
    std::string makeNdList(const std::string &name, int n) const;
    Ref<std::string> makeAccList(GenISLExprDeps &genISLExpr,
                                 const std::vector<Expr> &list, RelaxMode relax,
                                 ExternalMap &externals);
    Ref<std::string> makeCond(GenISLExprDeps &genISLExpr,
                              const std::vector<Expr> &conds, RelaxMode relax,
                              ExternalMap &externals);

    ISLMap makeAccMap(ISLCtx &isl, GenISLExprDeps &genISLExpr,
                      const AccessPoint &p, int iterDim, int accDim,
                      RelaxMode relax, const std::string &extSuffix,
                      ExternalMap &externals);

    ISLMap makeEqForBothOps(ISLCtx &isl,
                            const std::vector<std::pair<int, int>> &coord,
                            int iterDim) const;
    ISLMap makeIneqBetweenOps(ISLCtx &isl, DepDirection mode, int iterId,
                              int iterDim) const;

    ISLMap makeSerialToAll(ISLCtx &isl, int iterDim, int serialIterDim,
                           const std::vector<IterAxis> &point) const;
    static int countSerial(const std::vector<IterAxis> &point);

    ISLMap makeExternalEq(ISLCtx &isl, int iterDim, const std::string &ext1,
                          const std::string &ext2);

    ISLMap makeConstraintOfSingleLoop(ISLCtx &isl, const std::string &loop,
                                      DepDirection mode, int iterDim);

    ISLMap makeConstraintOfParallelScope(ISLCtx &isl,
                                         const std::string &parallel,
                                         DepDirection mode, int iterDim,
                                         const Ref<AccessPoint> &point,
                                         const Ref<AccessPoint> &other);

    /**
     * Constraint for variables defined inside some loops
     * E.g.
     * for i
     *   var def a
     *     a[0] = i
     *     ... = a[0]
     * There will be no dependencies of a[0] across i
     */
    ISLMap makeEraseVarDefConstraint(ISLCtx &isl, const Ref<AccessPoint> &point,
                                     int iterDim);

    /**
     * Constraint for loops that explicitly marked as no_deps by users
     */
    ISLMap makeNoDepsConstraint(ISLCtx &isl, const std::string &var,
                                int iterDim);

    /*
     * Constraint for external variables inside loop
     * E.g.
     * for i
     *   for j
     *     a[idx[i] + j]
     * idx[i] + j must be different for the same i but different j, but
     * idx[i] + j may be the same for different i
     */
    ISLMap makeExternalVarConstraint(ISLCtx &isl, const Ref<AccessPoint> &point,
                                     const Ref<AccessPoint> &other,
                                     const ExternalMap &pExternals,
                                     const ExternalMap &oExternals, int iterDim,
                                     const std::string &extSuffixP,
                                     const std::string &extSuffixO);

    /**
     * If we are analyzing the dependency between A and B, e.g.
     * for i
     *   for j
     *     A
     *   for k
     *     B
     * Analyzing the value of j and k will spend a great amount of time, but in
     * FindDepsMode::Dep mode, we do not care about the result. Therefore, we
     * project out these dimensions
     */
    ISLMap projectOutPrivateAxis(ISLCtx &isl, int iterDim, int since);

    int numCommonDims(const Ref<AccessPoint> &p1, const Ref<AccessPoint> &p2);

    static const std::string &getVar(const AST &op);

    /**
     * Check the dependencies between a later memory access `point` and many
     * earlier memory accesses in `otherList`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Earlier memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependency. All visitors to memory access nodes will call
     * `checkDep`
     */
    void checkDep(const Ref<AccessPoint> &point,
                  const std::vector<Ref<AccessPoint>> &otherList);
    void checkDepImpl(ISLCtx &isl, GenISLExprDeps &genISLExpr,
                      const Ref<AccessPoint> &point,
                      const std::vector<Ref<AccessPoint>> &otherList);

    template <class T> void visitStoreLike(const T &op) {
        Visitor::visit(op);
        auto &&point = points_.at(op);
        auto &&defId = defId_.at(op->var_);
        if (depType_ & DEP_WAR) {
            if (reads_.count(defId)) {
                checkDep(point, reads_.at(defId));
            }
        }
        if ((depType_ & DEP_WAW) ||
            ((depType_ & DEP_RAW) && op->nodeType() == ASTNodeType::ReduceTo)) {
            if (writes_.count(defId)) {
                std::vector<Ref<AccessPoint>> others;
                for (auto &&item : writes_.at(defId)) {
                    if (ignoreReductionWAW_ &&
                        op->nodeType() == ASTNodeType::ReduceTo &&
                        item->op_->nodeType() == ASTNodeType::ReduceTo) {
                        continue;
                    }
                    others.emplace_back(item);
                }
                checkDep(point, others);
            }
        }
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
    void visit(const MatMul &op) override { (*this)(op->equivalent_); }
};

/**
 * Find all dependencies of a specific type along the given loops
 *
 * @param op : AST root
 * @param cond : conditions to check: reduce_or [ reduce_and [ axis, mode ]]
 * @param found : callback
 * @param mode : Dep: all possible dependencies; Kill: all the situations that a
 * later access completely covers a earlier one
 * @param depType : WAW, RAW, RAW, or their combinations
 * @param filter : Additional callback to select which dependencies to check.
 * Return false in this callback to skip some dependencies. This callback can be
 * nullptr
 * @param ignoreReductionWAW : Ignore WAW dependencies between two ReduceTo
 * nodes. This kind of dependencies are false dependencies if running serially
 * @param eraseOutsideVarDef : Ignore all dependencies outside the VarDef
 */
void findDeps(const Stmt &op, const std::vector<FindDepsCond> &cond,
              const FindDepsCallback &found,
              FindDepsMode mode = FindDepsMode::Dep, DepType depType = DEP_ALL,
              const FindDepsFilter &filter = nullptr,
              bool ignoreReductionWAW = true, bool eraseOutsideVarDef = true);

inline std::string dep2Str(const NodeIDOrParallelScope &scope,
                           const std::string &var, const AST &later,
                           const AST &earlier) {
    std::ostringstream os;
    os << "Dependency "
       << (later->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ") << later
       << " after "
       << (earlier->nodeType() == ASTNodeType::Load ? "READ " : "WRITE ")
       << earlier << " along " << scope.name_ << " cannot be resolved";
    return std::regex_replace(os.str(), std::regex("\n"), "");
}

}; // namespace ir

#endif // DEPS_H
