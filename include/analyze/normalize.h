#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <mutator.h>

namespace ir {

/**
 * Hint nodes with extra info
 *
 * Please hide temporary info in a pass as much as possible. `Normalize` is not
 * the best way
 */
class Normalize : public Mutator {
  protected:
    Stmt visit(const For &op) override;
    Stmt visit(const If &op) override;
};

inline Stmt normalize(const Stmt &op) { return Normalize()(op); }

} // namespace ir

#endif // NORMALIZE_H
