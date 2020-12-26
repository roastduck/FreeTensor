#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <mutator.h>

namespace ir {

/**
 * Hint all If nodes with (x < 0) like conditions
 */
class NormalizeIf : public Mutator {
  protected:
    virtual Stmt visit(const If &op) override;
};

inline Stmt normalizeIf(const Stmt &op) { return NormalizeIf()(op); }

} // namespace ir

#endif // NORMALIZE_H
