#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>

namespace ir {

Sketch Sketch::genRandAnnotation() const {
    Sketch sketch = *this;
    for (auto &part : sketch.parts_) {
        part->genRandAnnotation();
    }
    return sketch;
}

Sketch::Sketch(const Schedule &schedule)
    : schedule_(schedule), annotated_(false) {}

void Sketch::addPart(const SketchPart &p) { parts_.push_back(p); }

Schedule Sketch::genSchedule() const {
    assert(annotated_);
    Schedule schedule = schedule_.clone();
    for (const auto &part : parts_)
        part->apply(schedule);
    return schedule;
}

bool Sketch::operator<(const Sketch &a) const { return time_ < a.time_; }

std::pair<bool, Sketch> Sketch::genMutation() const {
    Sketch ret = *this;
    int mut_part = random_int(ret.parts_.size() - 1);
    auto mut = ret.parts_[mut_part]->mutate();
    if (!mut.isValid()) {
        return std::make_pair(false, Sketch());
    }
    ret.parts_[mut_part] = mut;
    return std::make_pair(true, ret);
}

std::pair<bool, Sketch> Sketch::genCrossover(const Sketch &sketch) const {
    Sketch ret = *this;
    int mut_part = random_int(ret.parts_.size() - 1);
    auto mut = ret.parts_[mut_part]->crossover(sketch.parts_[mut_part]);
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

} // namespace ir
