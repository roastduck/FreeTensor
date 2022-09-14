#include <analyze/all_defs.h>
#include <analyze/all_stmts.h>
#include <analyze/all_uses.h>
#include <analyze/symbol_table.h>
#include <schedule.h>

namespace freetensor {

namespace {

class FindAllAccesses : public SymbolTable<Visitor> {
    typedef SymbolTable<Visitor> BaseClass;

    ID scopeToFind_;
    bool inside_ = false;

    // VarDef -> [[IDs of loops of iterators used] in each index] of each access
    std::unordered_map<VarDef, std::vector<std::vector<std::vector<ID>>>>
        accesses_;

    template <class T> void visitAcc(const T &op) {
        BaseClass::visit(op);
        if (inside_) {
            std::vector<std::vector<ID>> thisAccess;
            thisAccess.reserve(op->indices_.size());
            for (auto &&idx : op->indices_) {
                auto iters = allIters(idx, true);
                thisAccess.emplace_back(ranges::to<std::vector>(
                    iters | views::transform([this](const std::string &iter) {
                        return loop(iter)->id();
                    })));
            }
            accesses_[def(op->var_)].emplace_back(std::move(thisAccess));
        }
    }

  public:
    FindAllAccesses(const ID &scopeToFind) : scopeToFind_(scopeToFind) {}

    const auto &accesses() const { return accesses_; }

    void visitStmt(const Stmt &stmt) override {
        if (stmt->id() == scopeToFind_) {
            inside_ = true;
            BaseClass::visitStmt(stmt);
            inside_ = false;
        } else {
            BaseClass::visitStmt(stmt);
        }
    }

    using BaseClass::visit;
    void visit(const Load &op) override { visitAcc(op); }
    void visit(const Store &op) override { visitAcc(op); }
    void visit(const ReduceTo &op) override { visitAcc(op); }
};

} // Anonymous namespace

void Schedule::autoUseLib(const Target &target) {
    auto doUseLib = [&](const ID &toTry) {
        beginTransaction();
        try {
            // If some dimensions of a tensor is accessed by some iterators, and
            // these iterators are usually used to access some (maybe other)
            // tensors together, we prefer reordering these dimensions together.
            // E.g.:
            //
            // ```
            // c[i0, j0, i1, j1] += a[i0, k0, i1, k1] * b[k0, j0, k1, j1]
            // ```
            //
            // This will not accept by `asMatMul`. But we can find that `i0, i1`
            // and `j0, j1` and `k0, k1` are used together more commonly than
            // `i0, j0` or `j0, k0` etc., so we can reorder the statement like:
            //
            // ```
            // c[i0, i1, j0, j1] += a[i0, i1, k0, k1] * b[k0, k1, j0, j1]
            // ```

            // Check all accesses
            std::unordered_map<std::pair<ID, ID>, int>
                affinity; // (loop, loop) -> how likely are they together
            FindAllAccesses finder(toTry);
            finder(ast());
            for (auto &&[def, accesses] : finder.accesses()) {
                for (auto &&indices : accesses) {
                    size_t n = indices.size();
                    for (size_t i = 0; i + 1 < n; i++) {
                        for (size_t j = i + 1; j < n; j++) {
                            for (auto &&l : indices[i]) {
                                for (auto &&r : indices[j]) {
                                    affinity[std::make_pair(l, r)]++;
                                    affinity[std::make_pair(r, l)]++;
                                }
                            }
                        }
                    }
                }
            }

            // Try reordering dimensions
            for (auto &&[def, accesses] : finder.accesses()) {
                auto initPermu = ranges::to<std::vector>(views::ints(
                    0, (int)def->buffer_->tensor()->shape().size()));
                auto permu = initPermu;
                int bestScore = -1;
                std::vector<int> bestPermu;
                do {
                    int score = 0;
                    for (auto &&indices : accesses) {
                        for (size_t i = 0, n = indices.size(); i + 1 < n; i++) {
                            for (auto &&l : indices[permu[i]]) {
                                for (auto &&r : indices[permu[i + 1]]) {
                                    score += affinity[std::make_pair(l, r)];
                                }
                            }
                        }
                    }
                    if (score > bestScore) {
                        bestScore = score, bestPermu = permu;
                    }
                } while (std::next_permutation(permu.begin(), permu.end()));
                if (bestPermu != initPermu) {
                    beginTransaction();
                    try {
                        varReorder(def->id(), bestPermu);
                        commitTransaction();
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                        // no rethrow
                    }
                }
            }

            // Try asMatMul
            asMatMul(toTry);
            commitTransaction();
        } catch (const InvalidSchedule &e) {
            abortTransaction();
            throw;
        }
    };

    // Try to implement each top-level loops with lib calls
    for (auto &&_loop : findAll("<For><-(!<For><-)*-|")) {
        // Suppose the root node is not <For>. It should be <VarDef>
        auto loop = _loop.as<ForNode>();
        try {
            doUseLib(loop->id());
        } catch (const InvalidSchedule &e) {
            // If the loop is marked as preferLibs, we inline all local
            // variables, fission all the statments apart, and try applying to
            // each of them
            bool isPreferLibs = false;
            for (For l = loop;;) {
                if (l->property_->preferLibs_) {
                    isPreferLibs = true;
                    break;
                }
                Stmt body = l->body_;
                while (body->nodeType() == ASTNodeType::VarDef) {
                    body = body.as<VarDefNode>()->body_;
                }
                if (body->nodeType() != ASTNodeType::For) {
                    break;
                } else {
                    l = body.as<ForNode>();
                }
            }
            if (isPreferLibs) {
                for (auto &&[defId, name] :
                     allDefs(loop, {AccessType::Cache})) {
                    try {
                        inlining(defId);
                    } catch (const InvalidSchedule &e) {
                        // do nothing
                    }
                }
                auto stmts =
                    allStmts(loop, {ASTNodeType::Store, ASTNodeType::ReduceTo});
                for (auto &&[i, stmt] : views::enumerate(stmts)) {
                    beginTransaction();
                    try {
                        fission(loop->id(), FissionSide::Before, stmt->id(),
                                "." + toString(i), "");
                        auto libStmtId =
                            fission(loop->id(), FissionSide::After, stmt->id(),
                                    "." + toString(i) + ".lib", "")
                                .first.at(loop->id());
                        doUseLib(libStmtId);
                        commitTransaction();
                    } catch (const InvalidSchedule &e) {
                        abortTransaction();
                    }
                }
            }
        }
    }
}

} // namespace freetensor
