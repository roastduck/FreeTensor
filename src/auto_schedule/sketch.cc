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

Schedule Sketch::genSchedule() {
    if (scheduleGenerated_)
        return generatedSchedule_;
    generatedSchedule_ = schedule_.clone();
    for (auto &sub : subs_) {
        for (auto &part : sub.parts) {
            part.second->apply(generatedSchedule_, sub);
        }
    }
    scheduleGenerated_ = true;
    return generatedSchedule_;
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

const Func &Sketch::lowered(const Ref<Target> &target) {
    if (!lowered_.isValid()) {
        genSchedule();
        if (generatedSchedule_.ast().isValid()) {
            lowered_ = lower(generatedSchedule_.func(), target);
        }
    }
    return lowered_;
}

} // namespace freetensor
