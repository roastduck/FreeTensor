#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
#include <hash.h>

namespace ir {

Sketch Sketch::genRandAnnotation(std::default_random_engine &gen) const {
    Sketch sketch = clone();
    for (auto &part : sketch.parts_) {
        part->genRandAnnotation(gen);
    }
    return sketch;
}

void Sketch::addPart(const SketchPart &p) { parts_.push_back(p); }

Schedule Sketch::genSchedule(const std::vector<Ref<InitRule>> &rules) {
    if (scheduleGenerated_)
        return generatedSchedule_;
    generatedSchedule_ = schedule_.clone();
    for (auto &part : parts_) {
        part->apply(generatedSchedule_);
        for (auto &rule : rules) {
            rule->apply(part, generatedSchedule_);
        }
    }
    scheduleGenerated_ = true;
    return generatedSchedule_;
}

bool Sketch::operator<(const Sketch &a) const { return time_ < a.time_; }

std::pair<bool, Sketch>
Sketch::genMutation(std::default_random_engine &gen) const {
    Sketch ret = clone();
    int mut_part = randomInt(ret.parts_.size() - 1, gen);
    auto mut = ret.parts_[mut_part]->mutate(gen);
    if (!mut.isValid()) {
        return std::make_pair(false, Sketch());
    }
    ret.parts_[mut_part] = mut;
    return std::make_pair(true, ret);
}

std::pair<bool, Sketch>
Sketch::genCrossover(const Sketch &sketch,
                     std::default_random_engine &gen) const {
    Sketch ret = clone();
    int mut_part = randomInt(ret.parts_.size() - 1, gen);
    auto mut = ret.parts_[mut_part]->crossover(sketch.parts_[mut_part], gen);
    if (!mut.isValid()) {
        return std::make_pair(false, Sketch());
    }
    ret.parts_[mut_part] = mut;
    return std::make_pair(true, ret);
}

std::vector<int> Sketch::getAnnotation() const {
    std::vector<int> ret;
    for (const auto &part : parts_) {
        auto nw = part->getAnnotation();
        ret.insert(ret.end(), nw.begin(), nw.end());
    }
    return ret;
}

size_t Sketch::hash() const {
    size_t h = 0;
    for (const auto &part : parts_) {
        h = hashCombine(h, part->hash());
    }
    for (const auto &rule : doneRules_) {
        h = hashCombine(h, std::hash<std::string>{}(rule));
    }
    return h;
}

} // namespace ir
