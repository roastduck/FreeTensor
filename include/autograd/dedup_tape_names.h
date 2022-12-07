#ifndef FREE_TENSOR_DEDUP_TAPE_NAMES_H
#define FREE_TENSOR_DEDUP_TAPE_NAMES_H

#include <unordered_map>
#include <unordered_set>

#include <mutator.h>
#include <visitor.h>

namespace freetensor {

class CountNames : public Visitor {
    std::unordered_map<std::string, int> usedCnt_;

  public:
    const auto &usedCnt() const { return usedCnt_; }

  protected:
    void visit(const VarDef &op) override;
    void visit(const For &op) override;
};

class DedupTapeNames : public Mutator {
    const std::unordered_set<ID> &tapes_;
    const std::unordered_map<std::string, int> &usedCnt_;
    int dedupNumber_ = 0;

  public:
    DedupTapeNames(const std::unordered_set<ID> &tapes,
                   const std::unordered_map<std::string, int> &usedCnt)
        : tapes_(tapes), usedCnt_(usedCnt) {}

  protected:
    Stmt visit(const VarDef &op) override;
};

/**
 * Make all VarDef nodes in `tapes` having globally unique names
 */
Stmt dedupTapeNames(const Stmt &op, const std::unordered_set<ID> &tapes);

} // namespace freetensor

#endif // FREE_TENSOR_DEDUP_TAPE_NAMES_H
