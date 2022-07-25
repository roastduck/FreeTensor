#include <probability/rand_ctx.h>

namespace freetensor {

void RandCtxImpl::observeTrace(const Ref<RandTrace> &trace, double value,
                               double stddev) {
    auto i = traces_.emplace(trace, std::make_pair(value, stddev));

    size_t common;
    auto doOverserve = [&](const RandTrace &t0, double v0, double sigma0,
                           const RandTrace &t1, double v1, double sigma1) {
        while (common > 0 && (common > t0.size() || common > t1.size() ||
                              t1[common - 1] != t0[common - 1])) {
            // t[common - 1]: the last common item
            common--;
        }
        if (common < t0.size() && common < t1.size()) {
            // t[common]: the first non-common item
            if (v0 + sigma0 < v1 - sigma1) {
                t0[common].var_->observe(t0[common].value_);
            }
            if (v1 + sigma1 < v0 - sigma0) {
                t1[common].var_->observe(t1[common].value_);
            }
        }
    };

    common = trace->size();
    for (auto j = std::make_reverse_iterator(i); j != traces_.rend(); j++) {
        auto &&[t, v] = *j;
        doOverserve(*trace, value, stddev, *t, v.first, v.second);
    }

    common = trace->size();
    for (auto j = i; j != traces_.end(); j++) {
        auto &&[t, v] = *j;
        doOverserve(*trace, value, stddev, *t, v.first, v.second);
    }
}

} // namespace freetensor
