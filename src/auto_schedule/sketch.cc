#include <auto_schedule/sketch.h>
#include <auto_schedule/utils.h>
namespace ir {
Sketch Sketch::gen_rand_annotation() const {
    Sketch sketch = *this;
    for (auto &part : sketch.parts_) {
        part->gen_rand_annotation();
    }
    return sketch;
}

Sketch::Sketch(const Schedule &schedule)
    : schedule_(schedule), annotated(false) {}

void Sketch::add_part(const SketchPart &p) { parts_.push_back(p); }

Schedule Sketch::gen_schedule() const {
    assert(annotated);
    Schedule schedule = schedule_.clone();
    for (const auto &part : parts_)
        part->apply(schedule);
    return schedule;
}

bool Sketch::operator<(const Sketch &a) const { return time < a.time; }
std::pair<bool, Sketch> Sketch::gen_mutation() const {
    Sketch ret = *this;
    int mut_part = random_int(ret.parts_.size() - 1);
    auto mut = ret.parts_[mut_part]->mutate();
    if (!mut.isValid()) {
        return std::make_pair(false, Sketch());
    }
    ret.parts_[mut_part] = mut;
    return std::make_pair(true, ret);
}

std::pair<bool, Sketch> Sketch::gen_crossover(const Sketch &sketch) const {
    Sketch ret = *this;
    int mut_part = random_int(ret.parts_.size() - 1);
    auto mut = ret.parts_[mut_part]->crossover(sketch.parts_[mut_part]);
    if (!mut.isValid()) {
        return std::make_pair(false, Sketch());
    }
    ret.parts_[mut_part] = mut;
    return std::make_pair(true, ret);
}
std::vector<int> Sketch::get_annotation() const {
    std::vector<int> ret;
    for (const auto &part : parts_) {
        auto nw = part->get_annotation();
        ret.insert(ret.end(), nw.begin(), nw.end());
    }
    return ret;
}

} // namespace ir