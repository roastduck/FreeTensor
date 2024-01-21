#ifndef CUTLASS_MICRO_KERNEL_PROPERTY_H
#define CUTLASS_MICRO_KERNEL_PROPERTY_H

#include <expr.h>
#include <sub_tree.h>

namespace freetensor {

struct CutlassMicroKernelProperty : public ASTPart {
    int nWarpBatch_, nWarpM_, nWarpN_;
    Expr warpIdBatch_, warpIdM_, warpIdN_, laneId_;

    template <typename TwarpIdBatch, typename TwarpIdM, typename TwarpIdN,
              typename TlaneId>
    CutlassMicroKernelProperty(int nWarpBatch, int nWarpM, int nWarpN,
                               TwarpIdBatch &&warpIdBatch, TwarpIdM &&warpIdM,
                               TwarpIdN &&warpIdN, TlaneId &&laneId)
        : nWarpBatch_(nWarpBatch), nWarpM_(nWarpM), nWarpN_(nWarpN),
          warpIdBatch_(std::forward<TwarpIdM>(warpIdBatch)),
          warpIdM_(std::forward<TwarpIdM>(warpIdM)),
          warpIdN_(std::forward<TwarpIdN>(warpIdN)),
          laneId_(std::forward<TlaneId>(laneId)) {}

    void compHash() override;
};

inline Ref<CutlassMicroKernelProperty>
deepCopy(const Ref<CutlassMicroKernelProperty> &p) {
    return Ref<CutlassMicroKernelProperty>::make(
        p->nWarpBatch_, p->nWarpM_, p->nWarpN_, deepCopy(p->warpIdBatch_),
        deepCopy(p->warpIdM_), deepCopy(p->warpIdN_), deepCopy(p->laneId_));
}

} // namespace freetensor

#endif // CUTLASS_MICRO_KERNEL_PROPERTY_H
