#include <probability/rand_ctx.h>

namespace freetensor {

void RandCtxImpl::observeTrace(const Ref<RandTrace> &trace, double value,
                               double stddev) {
    auto i = traces_.emplace(trace, std::make_pair(value, stddev));

    size_t common;
    auto doOverserve = [&](const RandTrace &t0, double v0, double sigma0,
                           const RandTrace &t1, double v1, double sigma1) {
        // Maintain t[common] is the first non-common item
        while (common < t0.size() && common < t1.size() &&
               t0[common] == t1[common]) {
            common++;
        }
        if (common < t0.size() && common < t1.size()) {
            if (v0 + sigma0 < v1 - sigma1) {
                t0[common].var_->observe(t0[common].value_);
            }
            if (v1 + sigma1 < v0 - sigma0) {
                t1[common].var_->observe(t1[common].value_);
            }
        }
    };

    common = 0;
    for (auto j = traces_.begin(); j != i; j++) {
        auto &&[t, v] = *j;
        doOverserve(*trace, value, stddev, *t, v.first, v.second);
    }

    common = 0;
    for (auto j = traces_.rbegin(); j != std::make_reverse_iterator(i); j++) {
        auto &&[t, v] = *j;
        doOverserve(*trace, value, stddev, *t, v.first, v.second);
    }
}

} // namespace freetensor
