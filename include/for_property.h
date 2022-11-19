#ifndef FREE_TENSOR_FOR_PROPERTY_H
#define FREE_TENSOR_FOR_PROPERTY_H

#include <expr.h>
#include <parallel_scope.h>
#include <reduce_op.h>
#include <sub_tree.h>

namespace freetensor {

struct ReductionItem : public ASTPart {
    ReduceOp op_;
    std::string var_;
    SubTreeList<ExprNode> begins_ = ChildOf{this}, ends_ = ChildOf{this};

    void compHash() override;
};

template <class Tbegins, class Tends>
Ref<ReductionItem> makeReductionItem(ReduceOp op, const std::string &var,
                                     Tbegins &&begins, Tends &&ends) {
    auto r = Ref<ReductionItem>::make();
    r->op_ = op;
    r->var_ = var;
    r->begins_ = std::forward<Tbegins>(begins);
    r->ends_ = std::forward<Tends>(ends);
    return r;
}

struct ForProperty : public ASTPart {
    ParallelScope parallel_;
    bool unroll_, vectorize_;
    SubTreeList<ReductionItem> reductions_ = ChildOf{this};
    std::vector<std::string> noDeps_; // vars that are explicitly marked to have
                                      // no dependences over this loop
    bool preferLibs_; // Aggresively transform to external library calls in
                      // auto-schedule

    ForProperty()
        : parallel_(), unroll_(false), vectorize_(false), preferLibs_(false) {}

    Ref<ForProperty> withParallel(const ParallelScope &parallel) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->parallel_ = parallel;
        return ret;
    }
    Ref<ForProperty> withUnroll(bool unroll = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->unroll_ = unroll;
        return ret;
    }
    Ref<ForProperty> withVectorize(bool vectorize = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->vectorize_ = vectorize;
        return ret;
    }
    Ref<ForProperty> withNoDeps(const std::vector<std::string> &noDeps) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->noDeps_ = noDeps;
        return ret;
    }
    Ref<ForProperty> withPreferLibs(bool preferLibs = true) {
        auto ret = Ref<ForProperty>::make(*this);
        ret->preferLibs_ = preferLibs;
        return ret;
    }

    void compHash() override;
};

inline Ref<ReductionItem> deepCopy(const Ref<ReductionItem> &r) {
    return makeReductionItem(r->op_, r->var_, r->begins_, r->ends_);
}

inline Ref<ForProperty> deepCopy(const Ref<ForProperty> &_p) {
    auto p = Ref<ForProperty>::make();
    p->parallel_ = _p->parallel_;
    p->unroll_ = _p->unroll_;
    p->vectorize_ = _p->vectorize_;
    p->reductions_ = _p->reductions_;
    p->noDeps_ = _p->noDeps_;
    p->preferLibs_ = _p->preferLibs_;
    return p;
}

} // namespace freetensor

#endif // FREE_TENSOR_FOR_PROPERTY_H
