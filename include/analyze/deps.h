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
#include <analyze/find_stmt.h>
#include <analyze/symbol_table.h>
#include <analyze/track_stmt.h>
#include <container_utils.h>
#include <lazy.h>
#include <math/gen_pb_expr.h>
#include <math/presburger.h>
#include <serialize/to_string.h>
#include <sync_func.h>
#include <visitor.h>

namespace freetensor {

struct IterAxis {
    Expr iter_;
    ParallelScope parallel_;
    bool negStep_;

    IterAxis(Expr iter, const ParallelScope &parallel = serialScope,
             bool negStep = false)
        : iter_(iter), parallel_(parallel), negStep_(negStep) {}
};

struct Access {
    AST op_;
    Stmt stmt_;
    VarDef def_;
    Ref<Buffer> buffer_;
};

struct AccessPoint {
    AST op_;
    Stmt stmt_;
    VarDef def_;
    Ref<Buffer> buffer_;
    int defAxis_;                /// The position of the VarDef
    std::vector<IterAxis> iter_; /// The temporal location of the access
    std::vector<Expr> access_;   /// The spacial location of the access
    std::vector<std::pair<Expr, ID>>
        conds_; /// - first: The condition (predicate) of the access
                /// - second: the statement that contribute to the condition)
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

typedef SyncFunc<bool(const Access &)> FindDepsAccFilter;
typedef std::function<bool(const AccessPoint &)> FindDepsAccPtFilter;
typedef std::function<bool(const AccessPoint &later,
                           const AccessPoint &earlier)>
    FindDepsFilter;

/**
 * Find read and write points
 */
class FindAccessPoint : public SymbolTable<TrackStmt<Visitor>> {
    typedef SymbolTable<TrackStmt<Visitor>> BaseClass;

    ID vardef_;

    const FindDepsAccFilter &accFilter_;

    bool lastIsLoad_ =
        false; // True to indicate the last statement is a Load. If the next
               // statement is still a Load, they can share the same coordinate
    std::vector<IterAxis> cur_; // Current iteration point in the space
    std::vector<std::pair<Expr, ID>>
        conds_; // FIXME: There may be out-dated conditions, and we must check
                // allReads(cond) against allWrites(body) for each If or For
                // nodes. See pass/simplify. If the condition violates, we may
                // need to push a null condition according to RelaxMode
    std::vector<Ref<AccessPoint>> reads_, writes_;

    // For or StmtSeq -> coordinate in iteration space
    std::unordered_map<ID, std::vector<IterAxis>> scope2coord_;

    std::vector<ID> allScopes_;

    // Which axis is a the var defined
    int defAxis_ = -1;

  private:
    // Please use `doFind` instead
    using BaseClass::operator();

    void pushCond(const Expr &cond, const ID &baseStmtId) {
        conds_.emplace_back(cond, baseStmtId);
    }

    void popCond() { conds_.pop_back(); }

    /**
     * Check and remove trivial (1-lengthed) scope of StmtSeq
     *
     * @{
     */
    bool checkTrivialScope(std::vector<Ref<AccessPoint>>::iterator begin,
                           std::vector<Ref<AccessPoint>>::iterator end);
    void removeTrivialScopeFromAccesses(
        std::vector<Ref<AccessPoint>>::iterator begin,
        std::vector<Ref<AccessPoint>>::iterator end);
    void removeTrivialScopeFromScopes(std::vector<ID>::iterator begin,
                                      std::vector<ID>::iterator end);
    /** @} */

  public:
    FindAccessPoint(const ID &vardef, const FindDepsAccFilter &accFilter);

    void doFind(const Stmt &root);

    const auto &reads() const { return reads_; }
    const auto &writes() const { return writes_; }
    const auto &scope2coord() const { return scope2coord_; }

  private:
    template <class T> void visitStoreLike(const T &op) {
        BaseClass::visit(op);

        bool isThisVarDef = false;
        VarDef viewOf;
        if (def(op->var_)->id() == vardef_) {
            isThisVarDef = true;
        } else {
            for (auto source = def(op->var_); source->viewOf_.has_value();) {
                source = def(*source->viewOf_);
                if (source->id() == vardef_) {
                    isThisVarDef = true;
                    viewOf = source;
                    break;
                }
            }
        }
        if (!isThisVarDef) {
            return;
        }

        std::vector<Expr> exprs;
        VarDef d;
        if (viewOf.isValid()) {
            // Simultaneously access of a `VarDef` and the `VarDef` it views is
            // ALWAYS treated as dependences. Use Intrinsic as "any expression"
            // FIXME: This may cause problems if we want the presburger
            // expression from the `found_` callback. If `a[i]` is dependent on
            // `a[any]`, we want `i` to be really anything, instead of `i ==
            // this_intrin`
            exprs = std::vector<Expr>(
                viewOf->buffer_->tensor()->shape().size(),
                makeIntrinsic("", {}, DataType::Int32, false));
            d = viewOf;
        } else {
            exprs = op->indices_;
            d = def(op->var_);
        }

        if (accFilter_ == nullptr ||
            accFilter_(Access{op, curStmt(), d, d->buffer_})) {
            if (!cur_.empty() &&
                cur_.back().iter_->nodeType() == ASTNodeType::IntConst) {
                // top is band node
                cur_.back().iter_ = makeIntConst(
                    cur_.back().iter_.as<IntConstNode>()->val_ + 1);
            }
            lastIsLoad_ = false;

            auto ap = Ref<AccessPoint>::make();
            *ap = {op,   curStmt(),        d,     d->buffer_, defAxis_,
                   cur_, std::move(exprs), conds_};
            writes_.emplace_back(ap);
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

struct Dependence {
    const FindDepsDir &dir_; /// Direction vector filtering out this dependence
    const std::string &var_;
    const AccessPoint &later_, &earlier_;
    int iterDim_;

    // Some raw presburger objects. Note that the iteration space follows
    // temporal order, and an iterator `i` becomes `-i` when the loop traverses
    // reversedly
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

typedef SyncFunc<void(const Dependence &)> FindDepsCallback;

typedef int DepType;
const DepType DEP_WAW = 0x1;
const DepType DEP_WAR = 0x2;
const DepType DEP_RAW = 0x4;
const DepType DEP_ALL = DEP_WAW | DEP_WAR | DEP_RAW;

enum class RelaxMode : int { Possible, Necessary };
enum class FindDepsMode : int {
    Dep,         // Dependence may happen between `earlier` and `later`
    KillEarlier, // At any point in the space of `earlier`, it is dependent by
                 // `later`
    KillLater,   // At any point in the space of `later`, it is depending on
                 // `earlier`
    KillBoth,    // KillEarlier + KillLater
};

/**
 * Find RAW, WAR and WAW dependences
 */
class AnalyzeDeps {
    friend Dependence;

    std::vector<Ref<AccessPoint>> readsAsEarlier_, readsAsLater_,
        writesAsEarlier_, writesAsLater_;
    const std::unordered_map<ID, std::vector<IterAxis>> &scope2coord_;
    const std::unordered_map<std::string, std::vector<ID>>
        &noDepsLists_; // Var name -> [loop ID]
    Lazy<LoopVariExprMap> variantExpr_;

    const std::vector<FindDepsDir> &direction_;
    const FindDepsCallback &found_;
    const FindDepsAccPtFilter &earlierFilter_, &laterFilter_;
    const FindDepsFilter &filter_;

    const FindDepsMode mode_;
    const RelaxMode earlierRelax_, laterRelax_;
    const DepType depType_;
    const bool ignoreReductionWAW_;
    const bool eraseOutsideVarDef_;
    const bool noProjectOutPrivateAxis_;

    std::vector<std::function<void()>> tasks_;

  public:
    AnalyzeDeps(
        const std::vector<Ref<AccessPoint>> &reads,
        const std::vector<Ref<AccessPoint>> &writes,
        const std::unordered_map<ID, std::vector<IterAxis>> &scope2coord,
        const std::unordered_map<std::string, std::vector<ID>> &noDepsLists,
        const Lazy<LoopVariExprMap> &variantExpr,
        const std::vector<FindDepsDir> &direction,
        const FindDepsCallback &found, FindDepsMode mode, DepType depType,
        const FindDepsAccPtFilter &earlierFilter,
        const FindDepsAccPtFilter &laterFilter, const FindDepsFilter &filter,
        bool ignoreReductionWAW, bool eraseOutsideVarDef,
        bool noProjectOutPrivateAxis)
        : scope2coord_(scope2coord), noDepsLists_(noDepsLists),
          variantExpr_(variantExpr), direction_(direction), found_(found),
          earlierFilter_(earlierFilter), laterFilter_(laterFilter),
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
          eraseOutsideVarDef_(eraseOutsideVarDef),
          noProjectOutPrivateAxis_(noProjectOutPrivateAxis) {
        readsAsEarlier_ =
            ::freetensor::filter(reads, [&](const Ref<AccessPoint> &acc) {
                return earlierFilter_ == nullptr || earlierFilter_(*acc);
            });
        readsAsLater_ =
            ::freetensor::filter(reads, [&](const Ref<AccessPoint> &acc) {
                return laterFilter_ == nullptr || laterFilter_(*acc);
            });
        writesAsEarlier_ =
            ::freetensor::filter(writes, [&](const Ref<AccessPoint> &acc) {
                return earlierFilter_ == nullptr || earlierFilter_(*acc);
            });
        writesAsLater_ =
            ::freetensor::filter(writes, [&](const Ref<AccessPoint> &acc) {
                return laterFilter_ == nullptr || laterFilter_(*acc);
            });
    }

    void genTasks();

    const std::vector<std::function<void()>> &tasks() const { return tasks_; }

  public:
    static std::string makeIterList(const std::vector<IterAxis> &list, int n);
    static std::string makeNegIterMap(const std::vector<IterAxis> &list, int n);
    static std::string makeNdList(const std::string &name, int n);
    static std::string makeAccList(GenPBExpr &genPBExpr,
                                   const std::vector<Expr> &list,
                                   RelaxMode relax,
                                   GenPBExpr::VarMap &externals);
    static std::string makeCond(GenPBExpr &genPBExpr,
                                const std::vector<std::pair<Expr, ID>> &conds,
                                RelaxMode relax, GenPBExpr::VarMap &externals,
                                bool eraseOutsideVarDef, const VarDef &vardef);

  private:
    PBMap makeAccMap(PBCtx &presburger, const AccessPoint &p, int iterDim,
                     int accDim, RelaxMode relax, const std::string &extSuffix,
                     GenPBExpr::VarMap &externals,
                     const ASTHashSet<Expr> &noNeedToBeVars);

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
     *
     * E.g.
     *
     * ```
     * for i
     *   var def a
     *     a[0] = i
     *     ... = a[0]
     * ```
     *
     * There will be no dependences of a[0] across i
     */
    PBMap makeEraseVarDefConstraint(PBCtx &presburger,
                                    const Ref<AccessPoint> &point, int iterDim);

    /**
     * Constraint for loops that explicitly marked as no_deps by users
     */
    PBMap makeNoDepsConstraint(PBCtx &presburger, const std::string &var,
                               int iterDim);

    /**
     * Constraint for external variables inside loop
     *
     * E.g.
     *
     * ```
     * for i
     *   for j
     *     a[idx[i] + j]
     * ```
     *
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
     * If we are analyzing the dependence between A and B, e.g.
     *
     * ```
     * for i
     *   for j
     *     A
     *   for k
     *     B
     * ```
     *
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
                          const PBMap &earlierMap, const PBMap &extConstraint,
                          int iterDim);

    static const std::string &getVar(const AST &op);

    /**
     * Check the dependences between a later memory access `later` and many
     * earlier memory accesses in `earlierList`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Earlier memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependence. Used for RAW and WAW dependences
     */
    void
    checkDepLatestEarlier(const Ref<AccessPoint> &later,
                          const std::vector<Ref<AccessPoint>> &earlierList);
    void
    checkDepLatestEarlierImpl(PBCtx &presburger, const Ref<AccessPoint> &later,
                              const std::vector<Ref<AccessPoint>> &earlierList);

    /**
     * Check the dependences between many later memory access in `laterList`
     * and a earlier memory accesses `earlier`, filter them via the `filter_`
     * callback, and report then via the `found_` callback. Later memory
     * accesses may overwrite each other, and the overwritten ones will not
     * result in a dependence. Used for WAR dependences
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
    FindDepsAccPtFilter earlierFilter_ = nullptr;
    FindDepsAccPtFilter laterFilter_ = nullptr;
    FindDepsFilter filter_ = nullptr;
    std::function<void(const ID &,
                       const std::unordered_map<ID, std::vector<IterAxis>> &)>
        scope2CoordCallback_ = nullptr;
    bool ignoreReductionWAW_ = true;
    bool eraseOutsideVarDef_ = true;
    bool noProjectOutPrivateAxis_ = false;

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
     *
     * Note: killing test is insensitive to loop-invariant, which means there
     * will be false nagative
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
     *
     * Pass a normal function to be called sequentially, or pass a `SyncFunc`
     * object to deciede whether to be called in parallel
     *
     * @{
     */
    FindDeps filterAccess(const std::function<bool(const Access &)> &f) {
        return filterAccess(syncFunc(f));
    }
    FindDeps filterAccess(const FindDepsAccFilter &f) {
        FindDeps ret = *this;
        ret.accFilter_ =
            ret.accFilter_ == nullptr
                ? f
                : unsyncFunc([f0 = ret.accFilter_, f1 = f](const Access &acc) {
                      return f0(acc) && f1(acc);
                  });
        return ret;
    }
    /** @} */

    /**
     * Configure an additional callback to select the dependent (earlier) access
     * to check
     *
     * `filterEarlier` is perferred over `filter` for performance
     *
     * Defaults to no filter
     */
    FindDeps filterEarlier(const FindDepsAccPtFilter &f) {
        FindDeps ret = *this;
        ret.earlierFilter_ =
            ret.earlierFilter_ == nullptr
                ? f
                : [f0 = ret.earlierFilter_, f1 = f](const AccessPoint &acc) {
                      return f0(acc) && f1(acc);
                  };
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
    FindDeps filterLater(const FindDepsAccPtFilter &f) {
        FindDeps ret = *this;
        ret.laterFilter_ =
            ret.laterFilter_ == nullptr
                ? f
                : [f0 = ret.laterFilter_, f1 = f](const AccessPoint &acc) {
                      return f0(acc) && f1(acc);
                  };
        return ret;
    }

    /**
     * Configure an additional callback to select which dependences to check
     *
     * Please use `filterAccess`, `filterEarlier` or `filterLater` if possbile,
     * for better performance
     *
     * Defaults to no filter
     */
    FindDeps filter(const FindDepsFilter &f) {
        FindDeps ret = *this;
        ret.filter_ = f;
        ret.filter_ =
            ret.filter_ == nullptr
                ? f
                : [f0 = ret.filter_, f1 = f](const AccessPoint &later,
                                             const AccessPoint &earlier) {
                      return f0(later, earlier) && f1(later, earlier);
                  };
        return ret;
    }

    /**
     * Help function to analyze a sub-AST only
     */
    FindDeps filterSubAST(const ID &subAST) {
        return filterAccess([subAST](const Access &acc) {
            return acc.stmt_->ancestorById(subAST).isValid();
        });
    }

    /**
     * Ignore WAW dependences between two ReduceTo nodes. This kind of
     * dependences are false dependences if running serially
     *
     * Defaults to true
     */
    FindDeps ignoreReductionWAW(bool flag) {
        FindDeps ret = *this;
        ret.ignoreReductionWAW_ = flag;
        return ret;
    }

    /**
     * Ignore all dependences outside the VarDef
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
    FindDeps noProjectOutPrivateAxis(bool flag) {
        FindDeps ret = *this;
        ret.noProjectOutPrivateAxis_ = flag;
        return ret;
    }

    FindDeps scope2CoordCallback(
        std::function<void(
            const ID &, const std::unordered_map<ID, std::vector<IterAxis>> &)>
            callback) {
        FindDeps ret = *this;
        ret.scope2CoordCallback_ = callback;
        return ret;
    }

    /**
     * Run FindDeps, synchronized as default
     *
     * @param op : AST root
     * @param found : callback
     */
    void operator()(const Stmt &op,
                    const std::function<void(const Dependence &)> &found) {
        (*this)(op, syncFunc(found));
    }

    /**
     * Run FindDeps, with specified synchronize/unsynchronize
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

std::ostream &operator<<(std::ostream &os, const Dependence &dep);

}; // namespace freetensor

#endif // FREE_TENSOR_DEPS_H
