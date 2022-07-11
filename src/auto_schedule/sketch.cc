#include <analyze/fixed_length_feature.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen.h>
#include <hash.h>
#include <lower.h>

namespace freetensor {

Sketch Sketch::genRandAnnotation(RNG &gen) const {
    Sketch sketch = clone();
    for (auto &sub : sketch.subs_) {
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

std::pair<bool, Sketch> Sketch::genMutation(RNG &gen) const {
    Sketch ret = clone();
    int mutSub = randomInt(ret.subs_.size() - 1, gen);
    int mutPart = randomInt(ret.subs_[mutSub].parts.size() - 1, gen);
    auto mut = std::next(ret.subs_[mutSub].parts.begin(), mutPart)
                   ->second->mutate(gen);
    if (!mut) {
        return std::make_pair(false, Sketch());
    }
    return std::make_pair(true, ret);
}

std::pair<bool, Sketch> Sketch::genCrossover(const Sketch &sketch,
                                             RNG &gen) const {
    Sketch ret = clone();

    int mutSub = randomInt(ret.subs_.size() - 1, gen);
    if (!ret.subs_[mutSub].canCrossOver(sketch.subs_[mutSub])) {
        return std::make_pair(false, Sketch());
    }
    int mutPart = randomInt(ret.subs_[mutSub].parts.size() - 1, gen);
    auto mut =
        std::next(ret.subs_[mutSub].parts.begin(), mutPart)
            ->second->crossover(
                std::next(sketch.subs_[mutSub].parts.begin(), mutPart)->second,
                gen);
    if (!mut) {
        return std::make_pair(false, Sketch());
    }
    return std::make_pair(true, ret);
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

std::string Sketch::genCode(const Ref<Target> &target) {
    if (!code_.empty()) {
        return code_;
    }
    genSchedule();
    if (!generatedSchedule_.ast().isValid()) {
        return "";
    }
    lowered_ = lower(generatedSchedule_.func(), target);
    code_ = codeGen(lowered_, target);
    return code_;
}
std::vector<double> &Sketch::genFeature(const Ref<Target> &target) {
    if (feature_.empty()) {
        genCode(target);
        feature_ = fixedLengthFeature(lowered_->body_);
    }
    return feature_;
}

} // namespace freetensor
