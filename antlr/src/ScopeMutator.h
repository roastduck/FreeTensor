#ifndef SCOPE_MUTATOR_H_
#define SCOPE_MUTATOR_H_

#include "Mutator.h"

class FlattenMutator : public Mutator {
protected:
    std::shared_ptr<StmtNode> mutate(const StmtSeqNode *op) override;
};

class VarScopeMutator : public Mutator {
protected:
    std::shared_ptr<StmtNode> mutate(const StmtSeqNode *op) override;
};

class ScopeMutator {
public:
    std::shared_ptr<ProgramNode> operator()(const std::shared_ptr<ProgramNode> &op) {
        return varScope_(flatten_(op));
    }

private:
    FlattenMutator flatten_;
    VarScopeMutator varScope_;
};

#endif  // SCOPE_MUTATOR_H_
