#ifndef DEPS_H
#define DEPS_H

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <analyze/symbol_table.h>
#include <cursor.h>
#include <math/gen_pb_expr.h>
#include <math/presburger.h>
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
class FindAccessPoint : public SymbolTable<VisitorWithCursor> {
    typedef SymbolTable<VisitorWithCursor> BaseClass;

    bool lastIsLoad_ = false;
    std::vector<IterAxis> cur_; // Current iteration point in the space
    std::vector<Expr> conds_;   // FIXME: There may be out-dated conditions. See
                                // OutDatedRemover in pass/simplify
    std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> reads_,
        writes_;
    std::vector<VarDef> allDefs_;

    // For or StmtSeq -> coordinate in space
    std::unordered_map<std::string, std::vector<IterAxis>> scope2coord_;

    // Var name -> axis: Which axis is a local var defined
    std::unordered_map<std::string, int> defAxis_;

  public:
    FindAccessPoint(const Stmt &root);

    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> &
    reads() const {
        return reads_;
    }
    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>> &
    writes() const {
        return writes_;
    }
    const std::vector<VarDef> &allDefs() const { return allDefs_; }
    const std::unordered_map<std::string, std::vector<IterAxis>> &
    scope2coord() const {
        return scope2coord_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        BaseClass::visit(op);

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
               def(op->var_),
               def(op->var_)->buffer_,
               defAxis_.at(op->var_),
               cur_,
               std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
               conds_};
        writes_[def(op->var_)->id()].emplace_back(ap);
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
class AnalyzeDeps {
    const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
        &reads_, &writes_;
    const std::vector<VarDef> &allDefs_;
    const std::unordered_map<std::string, std::vector<IterAxis>> &scope2coord_;
    const std::unordered_map<std::string, std::vector<std::string>>
        &noDepsLists_; // Var name -> [loop ID]
    const LoopVariExprMap &variantExpr_;

    const std::vector<FindDepsCond> &cond_;
    const FindDepsCallback &found_;
    const FindDepsFilter &filter_;

    const FindDepsMode mode_;
    const RelaxMode earlierRelax_, laterRelax_;
    const DepType depType_;
    const bool ignoreReductionWAW_;
    const bool eraseOutsideVarDef_;

    std::vector<std::function<void()>> tasks_;
    std::mutex lock_;

  public:
    AnalyzeDeps(
        const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
            &reads,
        const std::unordered_map<std::string, std::vector<Ref<AccessPoint>>>
            &writes,
        const std::vector<VarDef> &allDefs,
        const std::unordered_map<std::string, std::vector<IterAxis>>
            &scope2coord,
        const std::unordered_map<std::string, std::vector<std::string>>
            &noDepsLists,
        const LoopVariExprMap &variantExpr,
        const std::vector<FindDepsCond> &cond, const FindDepsCallback &found,
        FindDepsMode mode, DepType depType, const FindDepsFilter &filter,
        bool ignoreReductionWAW, bool eraseOutsideVarDef)
        : reads_(reads), writes_(writes), allDefs_(allDefs),
          scope2coord_(scope2coord), noDepsLists_(noDepsLists),
          variantExpr_(variantExpr), cond_(cond), found_(found),
          filter_(filter), mode_(mode),
          earlierRelax_(mode_ == FindDepsMode::KillLater ||
                                mode_ == FindDepsMode::KillBoth
                            ? RelaxMode::Necessary
                            : RelaxMode::Possible),
          laterRelax_(mode_ == FindDepsMode::KillEarlier ||
                              mode_ == FindDepsMode::KillBoth
                          ? RelaxMode::Necessary
                          : RelaxMode::Possible),
          depType_(depType), ignoreReductionWAW_(ignoreReductionWAW),
          eraseOutsideVarDef_(eraseOutsideVarDef) {}

    void genTasks();

    const std::vector<std::function<void()>> &tasks() const { return tasks_; }

  private:
    std::string makeIterList(const std::vector<IterAxis> &list, int n);
    std::string makeNdList(const std::string &name, int n) const;
    Ref<std::string> makeAccList(GenPBExpr &genPBExpr,
                                 const std::vector<Expr> &list, RelaxMode relax,
                                 GenPBExpr::VarMap &externals);
    Ref<std::string> makeCond(GenPBExpr &genPBExpr,
                              const std::vector<Expr> &conds, RelaxMode relax,
                              GenPBExpr::VarMap &externals);

    PBMap makeAccMap(PBCtx &presburger, const AccessPoint &p, int iterDim,
                     int accDim, RelaxMode relax, const std::string &extSuffix,
                     GenPBExpr::VarMap &externals);

    PBMap makeEqForBothOps(PBCtx &presburger,
                           const std::vector<std::pair<int, int>> &coord,
                           int iterDim) const;
    PBMap makeIneqBetweenOps(PBCtx &presburger, DepDirection mode, int iterId,
                             int iterDim) const;

    PBMap makeSerialToAll(PBCtx &presburger, int iterDim,
                          const std::vector<IterAxis> &point) const;
    static int countSerial(const std::vector<IterAxis> &point);

    PBMap makeExternalEq(PBCtx &presburger, int iterDim,
                         const std::string &ext1, const std::string &ext2);

    PBMap makeConstraintOfSingleLoop(PBCtx &presburger, const std::string &loop,
                                     DepDirection mode, int iterDim);

    PBMap makeConstraintOfParallelScope(PBCtx &presburger,
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
    PBMap makeEraseVarDefConstraint(PBCtx &presburger,
                                    const Ref<AccessPoint> &point, int iterDim);

    /**
     * Constraint for loops that explicitly marked as no_deps by users
     */
    PBMap makeNoDepsConstraint(PBCtx &presburger, const std::string &var,
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
    PBMap makeExternalVarConstraint(PBCtx &presburger,
                                    const Ref<AccessPoint> &point,
                                    const Ref<AccessPoint> &other,
                                    const GenPBExpr::VarMap &pExternals,
                                    const GenPBExpr::VarMap &oExternals,
                                    int iterDim);

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
    PBMap projectOutPrivateAxis(PBCtx &presburger, int iterDim, int since);
    void projectOutPrivateAxis(PBCtx &presburger, const Ref<AccessPoint> &point,
                               const std::vector<Ref<AccessPoint>> &otherList,
                               PBMap &pmap, std::vector<PBMap> &omapList,
                               int iterDim);
    int numCommonDims(const Ref<AccessPoint> &p1, const Ref<AccessPoint> &p2);

    void checkAgainstCond(PBCtx &presburger, const Ref<AccessPoint> &point,
                          const Ref<AccessPoint> &other, const PBMap &depAll,
                          const PBMap &nearest, const PBSet &pIter,
                          const PBSet &oIter, int iterDim);

    static const std::string &getVar(const AST &op);

    /**
     * Check the dependencies between a later memory access `point` and many
     * earlier memory accesses in `otherList`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Earlier memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependency. Used for RAW and WAW dependencies
     */
    void checkDepLatestEarlier(const Ref<AccessPoint> &point,
                               const std::vector<Ref<AccessPoint>> &otherList);
    void
    checkDepLatestEarlierImpl(PBCtx &presburger, const Ref<AccessPoint> &point,
                              const std::vector<Ref<AccessPoint>> &otherList);

    /**
     * Check the dependencies between many later memory access in `pointList`
     * and a earlier memory accesses `other`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Later memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependency. Used for WAR dependencies
     */
    void checkDepEarliestLater(const std::vector<Ref<AccessPoint>> &pointList,
                               const Ref<AccessPoint> &other);
    void
    checkDepEarliestLaterImpl(PBCtx &presburger,
                              const std::vector<Ref<AccessPoint>> &pointList,
                              const Ref<AccessPoint> &other);
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

std::string toString(const Dependency &dep);

}; // namespace ir

#endif // DEPS_H
