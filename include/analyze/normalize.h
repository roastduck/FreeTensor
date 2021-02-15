#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <mutator.h>

namespace ir {

/**
 * Hint nodes with extra info. Currently only ForNode::info_len_
 */
class Normalize : public Mutator {
  protected:
    virtual Stmt visit(const For &op) override;
};

inline Stmt normalize(const Stmt &op) { return Normalize()(op); }

} // namespace ir

#endif // NORMALIZE_H
