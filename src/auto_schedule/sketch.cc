#include <analyze/fixed_length_feature.h>
#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <hash.h>
#include <lower.h>

namespace freetensor {

Sketch Sketch::genRandAnnotation(std::default_random_engine &gen) const {
    Sketch sketch = clone();
    for (auto &target : sketch.targets_) {
        for (auto &part : target.parts) {
            part.second->genRandAnnotation(gen);
        }
    }
    return sketch;
}

void Sketch::addPart(const SketchPart &p) {
    targets_[nowTargetNum_].parts.emplace(p->partType(), p);
}

Schedule Sketch::genSchedule() {
    if (scheduleGenerated_)
        return generatedSchedule_;
    generatedSchedule_ = schedule_.clone();
    for (auto &target : targets_) {
        for (auto &part : target.parts) {
            part.second->apply(generatedSchedule_, target);
        }
    }
    scheduleGenerated_ = true;
    return generatedSchedule_;
}

bool Sketch::operator<(const Sketch &a) const { return time_ < a.time_; }

std::pair<bool, Sketch>
Sketch::genMutation(std::default_random_engine &gen) const {
    Sketch ret = clone();
    int mutTarget = randomInt(ret.targets_.size() - 1, gen);
    int mutPart = randomInt(ret.targets_[mutTarget].parts.size() - 1, gen);
    auto mut = std::next(ret.targets_[mutTarget].parts.begin(), mutPart)
                   ->second->mutate(gen);
    if (!mut) {
        return std::make_pair(false, Sketch());
    }
    return std::make_pair(true, ret);
}

std::pair<bool, Sketch>
Sketch::genCrossover(const Sketch &sketch,
                     std::default_random_engine &gen) const {
    Sketch ret = clone();

    int mutTarget = randomInt(ret.targets_.size() - 1, gen);
    if (!ret.targets_[mutTarget].canCrossOver(sketch.targets_[mutTarget])) {
        return std::make_pair(false, Sketch());
    }
    int mutPart = randomInt(ret.targets_[mutTarget].parts.size() - 1, gen);
    auto mut =
        std::next(ret.targets_[mutTarget].parts.begin(), mutPart)
            ->second->crossover(
                std::next(sketch.targets_[mutTarget].parts.begin(), mutPart)
                    ->second,
                gen);
    if (!mut) {
        return std::make_pair(false, Sketch());
    }
    return std::make_pair(true, ret);
}

std::vector<int> Sketch::getAnnotation() const {
    std::vector<int> ret;
    for (const auto &target : targets_) {
        for (const auto &part : target.parts) {
            auto nw = part.second->getAnnotation();
            ret.insert(ret.end(), nw.begin(), nw.end());
        }
    }
    return ret;
}

size_t Sketch::hash() const {
    size_t h = 0;
    for (const auto &target : targets_) {
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
    if (target->type() == TargetType::GPU)
        code_ = codeGenCUDA(lowered_);
    else
        code_ = codeGenCPU(lowered_);
    std::cout << "lowered" << std::endl;
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
