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
#include <cursor.h>
#include <visitor.h>

namespace ir {

struct AccessPoint {
    AST op_;
    Cursor cursor_;
    int defAxis_;
    std::vector<Expr> iter_, begin_, end_, access_;
};

/**
 * Find read and write points
 */
class FindAccessPoint : public VisitorWithCursor {
    std::vector<Expr> cur_;         // Current iteration point in the space
    std::vector<Expr> begin_, end_; // Point range in the space
    std::unordered_map<AST, Ref<AccessPoint>> points_;
    std::unordered_multimap<std::string, Ref<AccessPoint>> reads_, writes_;

    // For or StmtSeq -> coordinate in space
    std::unordered_map<std::string, std::vector<Expr>> scope2coord_;

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
    const std::unordered_map<std::string, std::vector<Expr>> &
    scope2coord() const {
        return scope2coord_;
    }

  private:
    template <class T> void visitStoreLike(const T &op) {
        // For a[i] = a[i] + 1, write happens after read
        cur_.emplace_back(makeIntConst(0));
        begin_.emplace_back(makeIntConst(0));
        end_.emplace_back(makeIntConst(2));
        auto ap = Ref<AccessPoint>::make();
        *ap = {op,     cursor(), defAxis_.at(op->var_), cur_,
               begin_, end_,     op->indices_};
        points_.emplace(op, ap);
        writes_.emplace(op->var_, ap);

        cur_.back() = makeIntConst(1);
        Visitor::visit(op);
        cur_.pop_back();
        begin_.pop_back();
        end_.pop_back();
    }

  protected:
    void visit(const VarDef &op) override;
    void visit(const StmtSeq &op) override;
    void visit(const For &op) override;
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

enum class FindDepsMode : int {
    Normal,
    Same,
    Inv,
};

typedef std::vector<std::pair<std::string, FindDepsMode>> FindDepsCond;

typedef std::function<void(
    const std::vector<std::pair<std::string, FindDepsMode>> &,
    const std::string &, const AST &, const AST &, const Cursor &,
    const Cursor &)>
    FindDepsCallback;

/**
 * Find RAW, WAR and WAW dependencies
 */
class AnalyzeDeps : public Visitor {
    const std::unordered_map<AST, Ref<AccessPoint>> &points_;
    const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads_,
        &writes_;
    const std::unordered_map<std::string, std::vector<Expr>> &scope2coord_;
    const std::unordered_map<AST, LinearExpr> &linear_;

    const std::vector<FindDepsCond> &cond_;
    const FindDepsCallback &found_;

    bool ignoreReductionWAW_;

    isl_ctx *isl_;

  public:
    AnalyzeDeps(
        const std::unordered_map<AST, Ref<AccessPoint>> &points,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &reads,
        const std::unordered_multimap<std::string, Ref<AccessPoint>> &writes,
        const std::unordered_map<std::string, std::vector<Expr>> &scope2coord,
        const std::unordered_map<AST, LinearExpr> &linear,
        const std::vector<FindDepsCond> &cond, const FindDepsCallback &found,
        bool ignoreReductionWAW)
        : points_(points), reads_(reads), writes_(writes),
          scope2coord_(scope2coord), linear_(linear), cond_(cond),
          found_(found), ignoreReductionWAW_(ignoreReductionWAW) {
        isl_ = isl_ctx_alloc();
    }

    ~AnalyzeDeps() { isl_ctx_free(isl_); }

  private:
    std::string normalizeId(const std::string &id) const;
    Ref<std::string> linear2str(const LinearExpr &lin) const;
    std::string makeIterList(const std::vector<Expr> &list, int eraseBefore,
                             int n) const;
    std::string makeLinList(const std::vector<Ref<LinearExpr>> &list) const;
    std::string makeRange(const std::vector<Expr> &point,
                          const std::vector<Expr> &begin,
                          const std::vector<Expr> &end) const;
    std::string makeNdList(const std::string &name, int n) const;
    std::string makeAccMap(const AccessPoint &p, int iterDim, int accDim) const;
    std::string makeEqForBothOps(const std::vector<std::pair<int, int>> &coord,
                                 int iterDim) const;
    std::string makeIneqBetweenOps(FindDepsMode mode, int iterId,
                                   int iterDim) const;

    static const std::string &getVar(const AST &op);

    void checkDep(const AccessPoint &lhs, const AccessPoint &rhs);

    template <class T> void visitStoreLike(const T &op) {
        Visitor::visit(op);
        auto &&point = points_.at(op);
        auto range = reads_.equal_range(op->var_);
        for (auto i = range.first; i != range.second; i++) {
            checkDep(*point, *(i->second)); // WAR
        }
        range = writes_.equal_range(op->var_);
        for (auto i = range.first; i != range.second; i++) {
            if (ignoreReductionWAW_ && op->nodeType() != ASTNodeType::Store &&
                i->second->op_->nodeType() != ASTNodeType::Store) {
                // No dependency between reductions
                continue;
            }
            checkDep(*point, *(i->second)); // WAW
        }
    }

  protected:
    void visit(const Store &op) override { visitStoreLike(op); }
    void visit(const ReduceTo &op) override { visitStoreLike(op); }
    void visit(const Load &op) override;
};

/**
 * Find all inverse (negative) dependencies along the given loops
 *
 * @param op : AST root. The user should run the `disambiguous` pass before pass
 * it in
 * @param cond : conditions to check: reduce_and [ reduce_or [ axis, mode ]]
 * @param found : callback(sub-condition that fails, var name, later op, earlier
 * op, later cursor, earlier cursor)
 * @param ignoreReductionWAW : Ignore WAW dependencies between two ReduceTo
 * nodes. This kind of dependencies is false dependencies if running serially
 */
void findDeps(const Stmt &op, const std::vector<FindDepsCond> &cond,
              const FindDepsCallback &found, bool ignoreReductionWAW = true);

}; // namespace ir

#endif // DEPS_H
