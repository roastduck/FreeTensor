#ifndef DEPS_H
#define DEPS_H

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/options.h>
#include <isl/set.h>

#include <analyze/analyze_linear.h>
#include <cursor.h>
#include <visitor.h>

namespace ir {

struct IterAxis {
    Expr iter_, begin_, end_; /// begin_[i] <= iter_[i] < end_[i]

    IterAxis(Expr iter, Expr begin, Expr end)
        : iter_(iter), begin_(begin), end_(end) {}
};

struct AccessPoint {
    AST op_;
    Cursor cursor_;
    int defAxis_;                /// The position of the VarDef
    std::vector<IterAxis> iter_; /// The temporal location of the access
    std::vector<Expr> access_;   /// The spacial location of the access
    Expr cond_;                  /// The condition (predicate) of the access
};

/**
 * Find read and write points
 */
class FindAccessPoint : public VisitorWithCursor {
    std::vector<IterAxis> cur_; // Current iteration point in the space
    Expr cond_;
    std::unordered_map<AST, Ref<AccessPoint>> points_;
    std::unordered_multimap<std::string, Ref<AccessPoint>> reads_, writes_;

    // For or StmtSeq -> coordinate in space
    std::unordered_map<std::string, std::vector<IterAxis>> scope2coord_;

    // Var name -> axis: Which axis is a local var defined
    std::unordered_map<std::string, int> defAxis_;

  public:
    const std::unordered_map<AST, Ref<AccessPoint>> &points() const {
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
    const std::unordered_map<std::string, std::vector<IterAxis>> &
    scope2coord() const {
        return scope2coord_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        // For a[i] = a[i] + 1, write happens after read
        cur_.emplace_back(makeIntConst(0), makeIntConst(0), makeIntConst(2));
        auto ap = Ref<AccessPoint>::make();
        *ap = {op, cursor(), defAxis_.at(op->var_), cur_, op->indices_, cond_};
        points_.emplace(op, ap);
        writes_.emplace(op->var_, ap);

        cur_.back().iter_ = makeIntConst(1);
        Visitor::visit(op);
        cur_.pop_back();
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const If &op) override;
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

/**
 * Serialize expressions like "linear expressions concatenated with LAnd" to a
 * string
 *
 * ISL rejects non-affine expressions
 *
 * ISL also rejects non-contiguous sets, therefore here is no LOr or LNot
 */
class GenISLExpr {
    AnalyzeLinear analyzeLinear_;
    std::unordered_map<std::string, std::string> idCache_; // IR IDs -> ISL IDs
    std::unordered_set<std::string> idFlag_;               // ISL IDs

  private:
    Ref<std::string> linear2str(const LinearExpr<int> &lin);

  public:
    std::string normalizeId(const std::string &id);
    Ref<std::string> operator()(const Expr &op);
};

enum class DepDirection : int {
    Normal,
    Same,
    Inv,
};

typedef std::vector<std::pair<std::string, DepDirection>> FindDepsCond;

struct Dependency {
    const std::vector<std::pair<std::string, DepDirection>>
        &cond_; /// sub-condition that fails
    const std::string &var_;
    const AccessPoint &later_, &earlier_;

    // Helper functions
    const AST &later() const { return later_.op_; }
    const AST &earlier() const { return earlier_.op_; }
};
typedef std::function<void(const Dependency &)> FindDepsCallback;

typedef int DepType;
const DepType DEP_WAW = 0x1;
const DepType DEP_WAR = 0x2;
const DepType DEP_RAW = 0x4;
const DepType DEP_ALL = DEP_WAW | DEP_WAR | DEP_RAW;

enum class RelaxMode : int { Possible, Necessary };
enum class FindDepsMode : int { Dep, Kill };

typedef std::function<bool(const AccessPoint &later,
                           const AccessPoint &earlier)>
    FindDepsFilter;

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps : public Visitor {
    const std::unordered_map<AST, Ref<AccessPoint>> &points_;
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads_,
        &writes_;
    const std::unordered_map<std::string, std::vector<IterAxis>> &scope2coord_;
    GenISLExpr genISLExpr_;

    const std::vector<FindDepsCond> &cond_;
    const FindDepsCallback &found_;
    const FindDepsFilter &filter_;

    FindDepsMode mode_;
    DepType depType_;
    bool ignoreReductionWAW_;

    isl_ctx *isl_;

  public:
    AnalyzeDeps(
        const std::unordered_map<AST, Ref<AccessPoint>> &points,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &writes,
        const std::unordered_map<std::string, std::vector<IterAxis>>
            &scope2coord,
        const std::vector<FindDepsCond> &cond, const FindDepsCallback &found,
        FindDepsMode mode, DepType depType, const FindDepsFilter &filter,
        bool ignoreReductionWAW)
        : points_(points), reads_(reads), writes_(writes),
          scope2coord_(scope2coord), cond_(cond), found_(found),
          filter_(filter), mode_(mode), depType_(depType),
          ignoreReductionWAW_(ignoreReductionWAW) {
        isl_ = isl_ctx_alloc();
        isl_options_set_on_error(isl_, ISL_ON_ERROR_ABORT);
    }

    ~AnalyzeDeps() { isl_ctx_free(isl_); }

  private:
    std::string makeIterList(const std::vector<IterAxis> &list, int eraseBefore,
                             int n);
    Ref<std::string> makeAccList(const std::vector<Expr> &list,
                                 RelaxMode relax);
    Ref<std::string> makeRange(const std::vector<IterAxis> &point,
                               RelaxMode relax);
    Ref<std::string> makeCond(const Expr &cond, RelaxMode relax);
    Ref<std::string> makeAccMap(const AccessPoint &p, int iterDim, int accDim,
                                RelaxMode relax);

    std::string makeNdList(const std::string &name, int n) const;
    std::string makeEqForBothOps(const std::vector<std::pair<int, int>> &coord,
                                 int iterDim) const;
    std::string makeIneqBetweenOps(DepDirection mode, int iterId,
                                   int iterDim) const;

    static const std::string &getVar(const AST &op);

    void checkDep(const AccessPoint &lhs, const AccessPoint &rhs);

    template <class T> void visitStoreLike(const T &op) {
        Visitor::visit(op);
        auto &&point = points_.at(op);
        if (depType_ & DEP_WAR) {
            auto range = reads_.equal_range(op->var_);
            for (auto i = range.first; i != range.second; i++) {
                checkDep(*point, *(i->second));
            }
        }
        if (depType_ & DEP_WAW) {
            auto range = writes_.equal_range(op->var_);
            for (auto i = range.first; i != range.second; i++) {
                if (filter_ != nullptr && !filter_(*point, *(i->second))) {
                    continue;
                }
                if (ignoreReductionWAW_ &&
                    op->nodeType() == ASTNodeType::ReduceTo &&
                    i->second->op_->nodeType() == ASTNodeType::ReduceTo) {
                    continue;
                }
                checkDep(*point, *(i->second));
            }
        }
    }

  protected:
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

Stmt prepareFindDeps(const Stmt &op);

/**
 * Find all dependencies of a specific type along the given loops
 *
 * @param op : AST root. The user should run the `prepareFindDeps` pass before
 * pass it in
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
 */
void findDeps(const Stmt &op, const std::vector<FindDepsCond> &cond,
              const FindDepsCallback &found,
              FindDepsMode mode = FindDepsMode::Dep, DepType depType = DEP_ALL,
              const FindDepsFilter &filter = nullptr,
              bool ignoreReductionWAW = true);

}; // namespace ir

#endif // DEPS_H
