#ifndef FREE_TENSOR_DEPS_H
#define FREE_TENSOR_DEPS_H

#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <analyze/find_loop_variance.h>
#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <container_utils.h>
#include <lazy.h>
#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <serialize/to_string.h>
#include <visitor.h>

namespace freetensor {

struct IterAxis {
    Expr iter_;
    ParallelScope parallel_;

    IterAxis(Expr iter, const ParallelScope &parallel = serialScope)
        : iter_(iter), parallel_(parallel) {}
};

struct AccessPoint {
    AST op_;
    Stmt stmt_;
    VarDef def_;
    Ref<Buffer> buffer_;
    int defAxis_;                /// The position of the VarDef
    std::vector<IterAxis> iter_; /// The temporal location of the access
    std::vector<Expr> access_;   /// The spacial location of the access
    std::vector<Expr> conds_;    /// The condition (predicate) of the access
};

class FindAllNoDeps : public Visitor {
    std::unordered_map<std::string, std::vector<ID>>
        results_; // Var name -> [loop ID]
    // FIXME: Currently, we record a var name to loop ID relation, which is not
    // rigorous because there will be different vars with the same name.
    // Recording a VarDef ID to loop ID relation may be a better choice.
    // However, VarDef may be INSIDE a loop after pass/gpu/normalize_threads,
    // and we are unable to find the VarDef ID

  public:
    const std::unordered_map<std::string, std::vector<ID>> &results() const {
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

typedef std::function<bool(const AccessPoint &)> FindDepsAccFilter;
typedef std::function<bool(const AccessPoint &later,
                           const AccessPoint &earlier)>
    FindDepsFilter;

/**
 * Find read and write points
 */
class FindAccessPoint : public SymbolTable<TrackStmt<Visitor>> {
    typedef SymbolTable<TrackStmt<Visitor>> BaseClass;

    const FindDepsAccFilter &accFilter_;

    bool lastIsLoad_ = false;
    std::vector<IterAxis> cur_; // Current iteration point in the space
    std::vector<Expr>
        conds_; // FIXME: There may be out-dated conditions, and we must check
                // allReads(cond) against allWrites(body) for each If or For
                // nodes. See pass/simplify. If the condition violates, we may
                // need to push a null condition according to RelaxMode
    std::unordered_map<ID, std::vector<Ref<AccessPoint>>> reads_, writes_;
    std::vector<VarDef> allDefs_;

    // For or StmtSeq -> coordinate in space
    std::unordered_map<ID, std::vector<IterAxis>> scope2coord_;

    // Var name -> axis: Which axis is a local var defined
    std::unordered_map<std::string, int> defAxis_;

  public:
    FindAccessPoint(const Stmt &root, const FindDepsAccFilter &accFilter);

    const std::unordered_map<ID, std::vector<Ref<AccessPoint>>> &reads() const {
        return reads_;
    }
    const std::unordered_map<ID, std::vector<Ref<AccessPoint>>> &
    writes() const {
        return writes_;
    }
    const std::vector<VarDef> &allDefs() const { return allDefs_; }
    const std::unordered_map<ID, std::vector<IterAxis>> &scope2coord() const {
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
               curStmt(),
               def(op->var_),
               buffer(op->var_),
               defAxis_.at(op->var_),
               cur_,
               std::vector<Expr>{op->indices_.begin(), op->indices_.end()},
               conds_};
        if (accFilter_ == nullptr || accFilter_(*ap)) {
            writes_[def(op->var_)->id()].emplace_back(ap);
        }
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
    ID id_;
    ParallelScope parallel_;
    bool isNode_;

    NodeIDOrParallelScope(const ID &id) : id_(id), isNode_(true) {}
    NodeIDOrParallelScope(const ParallelScope &parallel)
        : parallel_(parallel), isNode_(false) {}
};

typedef std::vector<std::pair<NodeIDOrParallelScope, DepDirection>> FindDepsDir;

class AnalyzeDeps;

struct Dependency {
    const FindDepsDir &dir_; /// Direction vector filtering out this dependence
    const std::string &var_;
    const AccessPoint &later_, &earlier_;
    int iterDim_;
    PBMap later2EarlierIter_;
    PBMap laterIter2Idx_, earlierIter2Idx_;
    PBCtx &presburger_;
    AnalyzeDeps &self_;

    // Helper functions
    const AST &later() const { return later_.op_; }
    const AST &earlier() const { return earlier_.op_; }
    const VarDef &def() const { return earlier_.def_; }
    ID defId() const { return earlier_.def_->id(); }

    // Additional condition check. This check is not multi-thread, so please use
    // the `cond` parameter of `findDeps` instead, if possible
    PBMap extraCheck(PBMap dep, const NodeIDOrParallelScope &nodeOrParallel,
                     const DepDirection &dir) const;
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
    KillLater,   // At any point in the space of `later`, it is depending on
                 // `earlier`
    KillBoth,    // KillEarlier + KillLater
};

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps {
    friend Dependency;

    std::unordered_map<ID, std::vector<Ref<AccessPoint>>> readsAsEarlier_,
        readsAsLater_, writesAsEarlier_, writesAsLater_;
    const std::vector<VarDef> &allDefs_;
    const std::unordered_map<ID, std::vector<IterAxis>> &scope2coord_;
    const std::unordered_map<std::string, std::vector<ID>>
        &noDepsLists_; // Var name -> [loop ID]
    Lazy<LoopVariExprMap> variantExpr_;

    const std::vector<FindDepsDir> &direction_;
    const FindDepsCallback &found_;
    const FindDepsAccFilter &earlierFilter_, &laterFilter_;
    const FindDepsFilter &filter_;

    const FindDepsMode mode_;
    const RelaxMode earlierRelax_, laterRelax_;
    const DepType depType_;
    const bool ignoreReductionWAW_;
    const bool eraseOutsideVarDef_;
    const bool noProjectOutProvateAxis_;

    std::vector<std::function<void()>> tasks_;
    std::mutex lock_;

  public:
    AnalyzeDeps(
        const std::unordered_map<ID, std::vector<Ref<AccessPoint>>> &reads,
        const std::unordered_map<ID, std::vector<Ref<AccessPoint>>> &writes,
        const std::vector<VarDef> &allDefs,
        const std::unordered_map<ID, std::vector<IterAxis>> &scope2coord,
        const std::unordered_map<std::string, std::vector<ID>> &noDepsLists,
        Lazy<LoopVariExprMap> variantExpr,
        const std::vector<FindDepsDir> &direction,
        const FindDepsCallback &found, FindDepsMode mode, DepType depType,
        const FindDepsAccFilter &earlierFilter,
        const FindDepsAccFilter &laterFilter, const FindDepsFilter &filter,
        bool ignoreReductionWAW, bool eraseOutsideVarDef,
        bool noProjectOutProvateAxis)
        : allDefs_(allDefs), scope2coord_(scope2coord),
          noDepsLists_(noDepsLists), variantExpr_(variantExpr),
          direction_(direction), found_(found), earlierFilter_(earlierFilter),
          laterFilter_(laterFilter), filter_(filter), mode_(mode),
          earlierRelax_(mode_ == FindDepsMode::KillLater ||
                                mode_ == FindDepsMode::KillBoth
                            ? RelaxMode::Necessary
                            : RelaxMode::Possible),
          laterRelax_(mode_ == FindDepsMode::KillEarlier ||
                              mode_ == FindDepsMode::KillBoth
                          ? RelaxMode::Necessary
                          : RelaxMode::Possible),
          depType_(depType), ignoreReductionWAW_(ignoreReductionWAW),
          eraseOutsideVarDef_(eraseOutsideVarDef),
          noProjectOutProvateAxis_(noProjectOutProvateAxis) {
        for (auto &&[id, list] : reads) {
            readsAsEarlier_[id] =
                ::freetensor::filter(list, [&](const Ref<AccessPoint> &acc) {
                    return earlierFilter_ == nullptr || earlierFilter_(*acc);
                });
            readsAsLater_[id] =
                ::freetensor::filter(list, [&](const Ref<AccessPoint> &acc) {
                    return laterFilter_ == nullptr || laterFilter_(*acc);
                });
        }
        for (auto &&[id, list] : writes) {
            writesAsEarlier_[id] =
                ::freetensor::filter(list, [&](const Ref<AccessPoint> &acc) {
                    return earlierFilter_ == nullptr || earlierFilter_(*acc);
                });
            writesAsLater_[id] =
                ::freetensor::filter(list, [&](const Ref<AccessPoint> &acc) {
                    return laterFilter_ == nullptr || laterFilter_(*acc);
                });
        }
    }

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

    PBMap makeConstraintOfSingleLoop(PBCtx &presburger, const ID &loop,
                                     DepDirection mode, int iterDim);

    PBMap makeConstraintOfParallelScope(PBCtx &presburger,
                                        const ParallelScope &parallel,
                                        DepDirection mode, int iterDim,
                                        const AccessPoint &later,
                                        const AccessPoint &earlier);

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
                                    const Ref<AccessPoint> &later,
                                    const Ref<AccessPoint> &earlier,
                                    const GenPBExpr::VarMap &laterExternals,
                                    const GenPBExpr::VarMap &earlierExternals,
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
                               std::vector<PBMap> &otherMapList, int iterDim);
    int numCommonDims(const Ref<AccessPoint> &p1, const Ref<AccessPoint> &p2);

    void checkAgainstCond(PBCtx &presburger, const Ref<AccessPoint> &later,
                          const Ref<AccessPoint> &earlier, const PBMap &depAll,
                          const PBMap &nearest, const PBMap &laterMap,
                          const PBMap &earlierMap, int iterDim);

    static const std::string &getVar(const AST &op);

    /**
     * Check the dependencies between a later memory access `later` and many
     * earlier memory accesses in `earlierList`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Earlier memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependency. Used for RAW and WAW dependencies
     */
    void
    checkDepLatestEarlier(const Ref<AccessPoint> &later,
                          const std::vector<Ref<AccessPoint>> &earlierList);
    void
    checkDepLatestEarlierImpl(PBCtx &presburger, const Ref<AccessPoint> &later,
                              const std::vector<Ref<AccessPoint>> &earlierList);

    /**
     * Check the dependencies between many later memory access in `laterList`
     * and a earlier memory accesses `earlier`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Later memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependency. Used for WAR dependencies
     */
    void checkDepEarliestLater(const std::vector<Ref<AccessPoint>> &laterList,
                               const Ref<AccessPoint> &earlier);
    void
    checkDepEarliestLaterImpl(PBCtx &presburger,
                              const std::vector<Ref<AccessPoint>> &laterList,
                              const Ref<AccessPoint> &earlier);
};

/**
 * Find dependences in an AST satisfiying given conditions
 *
 * Conditions can be set with member functions, and finally FindDeps can be run
 * via `operator()`, e.g. `FindDeps().direction(...).filter(...)(...)`
 */
class FindDeps {
    FindDepsMode mode_ = FindDepsMode::Dep;
    DepType type_ = DEP_ALL;
    std::vector<FindDepsDir> direction_ = {{}};
    FindDepsAccFilter accFilter_ = nullptr;
    FindDepsAccFilter earlierFilter_ = nullptr;
    FindDepsAccFilter laterFilter_ = nullptr;
    FindDepsFilter filter_ = nullptr;
    bool ignoreReductionWAW_ = true;
    bool eraseOutsideVarDef_ = true;
    bool noProjectOutProvateAxis_ = false;

  public:
    /**
     * Configure whether one access should depending on / be dependent by ALL
     * INSTANCES of another access
     *
     * Possible values are:
     *
     * - Dep: No restriction
     * - KillEarlier: Any instance of the `earlier` statement / expression is
     * dependent by `later`
     * - KillLater: Any instance of the `later` statement is depending on
     * `earlier`
     * - KillBoth: KillEarlier + KillLater
     *
     * Defaults to no restriction
     */
    FindDeps mode(FindDepsMode m) {
        FindDeps ret = *this;
        ret.mode_ = m;
        return ret;
    }

    /**
     * Check only for WAW, RAW and / or RAW dependences
     *
     * Defaults to no restriction
     */
    FindDeps type(DepType t) {
        FindDeps ret = *this;
        ret.type_ = t;
        return ret;
    }

    /**
     * Check only for given directions on loops or parallel scopes
     *
     * The direction array is in `reduce_or [ reduce_and [ axis, mode ]]`
     * format.
     *
     * E.g. 1, `{{{L1, Same}, {L2, Normal}}}` means dependences should
     * happen inside one iteration of L1, AND happen along L2.
     *
     * E.g. 2, `{{{L1, Same}}, {{L2, Normal}}}` means dependences should
     * happen inside one iteration of L1, OR happen along L2.
     *
     * Defaults to no restriction
     */
    FindDeps direction(const std::vector<FindDepsDir> &d) {
        FindDeps ret = *this;
        ret.direction_ = d;
        return ret;
    }

    /**
     * Configure an additional callback to select the accesses to check
     *
     * `filterAccess` is preferred over `filterEarlier`, `filterLater` and
     * `filter` for performance
     *
     * Defaults to no filter
     */
    FindDeps filterAccess(const FindDepsAccFilter &f) {
        FindDeps ret = *this;
        ret.accFilter_ = f;
        return ret;
    }

    /**
     * Configure an additional callback to select the dependent (earlier) access
     * to check
     *
     * `filterEarlier` is perferred over `filter` for performance
     *
     * Defaults to no filter
     */
    FindDeps filterEarlier(const FindDepsAccFilter &f) {
        FindDeps ret = *this;
        ret.earlierFilter_ = f;
        return ret;
    }

    /**
     * Configure an additional callback to select the depending (later) access
     * to check
     *
     * `filterLater` is perferred over `filter` for performance
     *
     * Defaults to no filter
     */
    FindDeps filterLater(const FindDepsAccFilter &f) {
        FindDeps ret = *this;
        ret.laterFilter_ = f;
        return ret;
    }

    /**
     * Configure an additional callback to select which dependencies to check
     *
     * Please use `filterAccess`, `filterEarlier` or `filterLater` if possbile,
     * for better performance
     *
     * Defaults to no filter
     */
    FindDeps filter(const FindDepsFilter &f) {
        FindDeps ret = *this;
        ret.filter_ = f;
        return ret;
    }

    /**
     * Ignore WAW dependencies between two ReduceTo nodes. This kind of
     * dependencies are false dependencies if running serially
     *
     * Defaults to true
     */
    FindDeps ignoreReductionWAW(bool flag) {
        FindDeps ret = *this;
        ret.ignoreReductionWAW_ = flag;
        return ret;
    }

    /**
     * Ignore all dependencies outside the VarDef
     *
     * Defaults to true
     */
    FindDeps eraseOutsideVarDef(bool flag) {
        FindDeps ret = *this;
        ret.eraseOutsideVarDef_ = flag;
        return ret;
    }

    /**
     * Disable the projectOutPrivateAxis optimization. If you want to further
     * check Presburger maps or sets in the `found` callback, you must set it to
     * true
     *
     * Defaults to false
     */
    FindDeps noProjectOutProvateAxis(bool flag) {
        FindDeps ret = *this;
        ret.noProjectOutProvateAxis_ = flag;
        return ret;
    }

    /**
     * Run FindDeps
     *
     * @param op : AST root
     * @param found : callback
     */
    void operator()(const Stmt &op, const FindDepsCallback &found);

    /**
     * Helper function to run FindDeps
     *
     * Only to check whether there is a dependence satisfying given conditions,
     * but not cared about what dependence it is
     *
     * @param op : AST root
     */
    bool exists(const Stmt &op);
};

std::ostream &operator<<(std::ostream &os, const Dependency &dep);

}; // namespace freetensor

#endif // FREE_TENSOR_DEPS_H
