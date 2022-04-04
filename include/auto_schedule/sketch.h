#ifndef IR_SKETCH_H
#define IR_SKETCH_H

#include <analyze/find_multi_level_tiling.h>
#include <random>
#include <schedule.h>
#include <utility>
#include <vector>

namespace ir {

class SketchPartNode;
class InitRule;

typedef Ref<SketchPartNode> SketchPart;

enum class SketchPartType : int {
    MultiLevelTiling,
    MultiLevelTilingWithFusion
};

class SketchPartNode {
  public:
    virtual void genRandAnnotation(std::default_random_engine &gen) = 0;
    virtual SketchPart mutate(std::default_random_engine &gen) {
        return nullptr;
    }
    virtual SketchPart crossover(const SketchPart &part,
                                 std::default_random_engine &gen) {
        return nullptr;
    };
    virtual void apply(Schedule &schedule) = 0;
    virtual SketchPartType partType() = 0;
    virtual std::vector<int> getAnnotation() const = 0;
    virtual ~SketchPartNode() = default;
    virtual size_t hash() const = 0;
};

class Sketch {
    Schedule schedule_;
    std::vector<std::string> doneRules_;
    std::vector<SketchPart> parts_;
    std::vector<ForsWithDataReuse> targets_;
    int nowTargetNum_;
    double time_;
    bool scheduleGenerated_{false};

  public:
    Sketch() = default;
    Sketch(const Sketch &) = default;
    Sketch(Schedule schedule, std::vector<ForsWithDataReuse> targets)
        : schedule_(schedule.clone()), targets_(std::move(targets)),
          nowTargetNum_(targets_.size() - 1) {}

    Sketch clone() const {
        Sketch ret = *this;
        ret.schedule_ = schedule_.clone();
        return ret;
    }

    Sketch genRandAnnotation(std::default_random_engine &gen) const;

    Schedule genSchedule(const std::vector<Ref<InitRule>> &rules);

    void addPart(const SketchPart &);
    SketchPart part(int i) { return parts_[i]; }

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] std::pair<bool, Sketch>
    genMutation(std::default_random_engine &gen) const;

    [[nodiscard]] std::pair<bool, Sketch>
    genCrossover(const Sketch &sketch, std::default_random_engine &gen) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }

    size_t hash() const;

    ForsWithDataReuse &nowTarget() { return targets_[nowTargetNum_]; }
    [[nodiscard]] const ForsWithDataReuse &nowTarget() const {
        return targets_[nowTargetNum_];
    }

    void moveToNextTarget() { nowTargetNum_--; }

    int nowTargetNum() const { return nowTargetNum_; }

    void addDoneRule(std::string name) { doneRules_.push_back(name); }

    Schedule &schedule() { return schedule_; }
    const Schedule &schedule() const { return schedule_; }
};

} // namespace ir

#endif // IR_SKETCH_H
