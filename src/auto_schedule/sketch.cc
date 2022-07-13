#include <analyze/fixed_length_feature.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
#include <hash.h>
#include <lower.h>

namespace freetensor {

Ref<Sketch> Sketch::genRandAnnotation(RNG &gen) const {
    auto sketch = clone();
    for (auto &sub : sketch->subs_) {
        for (auto &part : sub.parts) {
            part.second->genRandAnnotation(gen);
        }
    }
    return sketch;
}

void Sketch::addPart(const SketchPart &p) {
    subs_[nowSubNum_].parts.emplace(p->partType(), p);
}

const Schedule &Sketch::genSchedule() {
    if (genSchedule_.isValid())
        return *genSchedule_;
    genSchedule_ = Opt<Schedule>::make(schedule_.clone());
    for (auto &sub : subs_) {
        for (auto &part : sub.parts) {
            part.second->apply(*genSchedule_, sub);
        }
    }
    return *genSchedule_;
}

bool Sketch::operator<(const Sketch &a) const { return time_ < a.time_; }

Ref<Sketch> Sketch::genMutation(RNG &gen) const {
    Ref<Sketch> ret = clone();
    int mutSub = randomInt(ret->subs_.size() - 1, gen);
    int mutPart = randomInt(ret->subs_[mutSub].parts.size() - 1, gen);
    auto mut = std::next(ret->subs_[mutSub].parts.begin(), mutPart)
                   ->second->mutate(gen);
    return mut ? ret : nullptr;
}

Ref<Sketch> Sketch::genCrossover(const Sketch &sketch, RNG &gen) const {
    Ref<Sketch> ret = clone();

    int mutSub = randomInt(ret->subs_.size() - 1, gen);
    if (!ret->subs_[mutSub].canCrossOver(sketch.subs_[mutSub])) {
        return nullptr;
    }
    int mutPart = randomInt(ret->subs_[mutSub].parts.size() - 1, gen);
    auto mut =
        std::next(ret->subs_[mutSub].parts.begin(), mutPart)
            ->second->crossover(
                std::next(sketch.subs_[mutSub].parts.begin(), mutPart)->second,
                gen);
    return mut ? ret : nullptr;
}

std::vector<int> Sketch::getAnnotation() const {
    std::vector<int> ret;
    for (const auto &sub : subs_) {
        for (const auto &part : sub.parts) {
            auto nw = part.second->getAnnotation();
            ret.insert(ret.end(), nw.begin(), nw.end());
        }
    }
    return ret;
}

size_t Sketch::hash() const {
    size_t h = 0;
    for (const auto &target : subs_) {
        h = hashCombine(h, target.hash());
    }
    return h;
}

const Func &Sketch::lowered() {
    if (!lowered_.isValid()) {
        lowered_ = lower(genSchedule().func(), target_);
    }
    return lowered_;
}

const std::vector<double> &Sketch::feature() {
    if (!feature_.isValid()) {
        feature_ =
            Opt<std::vector<double>>::make(fixedLengthFeature(lowered()));
    }
    return *feature_;
}

} // namespace freetensor
