#ifndef ALL_READS_H
#define ALL_READS_H

#include <unordered_set>

#include <visitor.h>

namespace ir {

/**
 * Record all buffers that are read in an AST
 */
class AllReads : public Visitor {
    std::unordered_set<std::string> reads_;

  public:
    const std::unordered_set<std::string> &reads() const { return reads_; }

  protected:
    void visit(const Load &op) override;
};

std::unordered_set<std::string> allReads(const AST &op);

} // namespace ir

#endif // ALL_READS_H
