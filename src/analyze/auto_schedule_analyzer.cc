#include <analyze/auto_schedule_analyzer.h>
#include <analyze/all_reads.h>
#include <hash.h>
namespace ir {

void AutoScheduleAnalyzer::visit(const Store &op) {
    nowStore_ = op;
    BaseClass::visit(op);
    nowStore_ = nullptr;
}

StoreMap AutoScheduleAnalyzer::getConsumersOf(const std::string &name) {
    return reads_[name];
}

StoreMap AutoScheduleAnalyzer::getProducersOf(const std::string &name) {
    return writes_[name];
}

bool AutoScheduleAnalyzer::isElementWise(const Store &st, const Load &ld) {
    const auto &destShape = buffer(st->var_)->tensor().shape();
    const auto &srcShape = buffer(ld->var_)->tensor().shape();
    if (destShape.size() != srcShape.size()) {
        return false;
    }
    HashComparator comp;
    for (int i = 0; i < destShape.size(); i++) {
        if (!comp(destShape[i], srcShape[i])) {
            return false;
        }
    }
    for (int i = 0; i < st->indices_.size(); i++) {
        if (!comp(st->indices_[i], ld->indices_[i])) {
            return false;
        }
    }
    return true;
}

void AutoScheduleAnalyzer::visit(const Load &op) {
    if (nowStore_.isValid()) {
        reads_[nowStore_->var_].insert({op->var_, {nowStore_, op}});
        writes_[op->var_].insert({nowStore_->var_, {nowStore_, op}});
    }
    BaseClass::visit(op);
}

}  // namespace ir