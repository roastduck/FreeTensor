#ifndef FREE_TENSOR_SKETCH_H
#define FREE_TENSOR_SKETCH_H

#include <analyze/find_multi_level_tiling.h>
#include <map>
#include <random>
#include <schedule.h>
#include <utility>
#include <vector>

namespace freetensor {

class SketchPartNode;

typedef Ref<SketchPartNode> SketchPart;

enum class SketchPartType : int {
    MultiLevelTiling = 0,
    MultiLevelTilingWithFusion = 1,
    ThreadBind = 2,
};

struct SketchTarget;

class SketchPartNode {
  public:
    virtual void genRandAnnotation(std::default_random_engine &gen) = 0;
    virtual bool mutate(std::default_random_engine &gen) { return false; }
    virtual bool crossover(const SketchPart &part,
                           std::default_random_engine &gen) {
        return false;
    };
    virtual void apply(Schedule &schedule, SketchTarget &target) = 0;
    virtual SketchPartType partType() = 0;
    virtual std::vector<int> getAnnotation() const = 0;
    virtual ~SketchPartNode() = default;
    virtual size_t hash() const = 0;
    virtual SketchPart clone() const = 0;
};

typedef std::map<SketchPartType, SketchPart> PartMap;

struct SketchTarget {
    ForsWithDataReuse target;
    PartMap parts;
    std::string log;
    explicit SketchTarget(ForsWithDataReuse target)
        : target(std::move(target)) {}
    [[nodiscard]] bool canCrossOver(const SketchTarget &other) const {
        return log == other.log;
    }

    SketchTarget(const SketchTarget &other)
        : target(other.target), log(other.log) {
        for (auto &&[type, part] : other.parts) {
            parts.emplace(type, part->clone());
        }
    }

    [[nodiscard]] bool hasPart(SketchPartType tp) const {
        return parts.count(tp);
    }

    SketchPart getPart(SketchPartType tp) {
        if (hasPart(tp)) {
            return parts[tp];
        }
        return nullptr;
    }

    [[nodiscard]] size_t hash() const {
        size_t h = hashCombine(std::hash<ForsWithDataReuse>{}(target),
                               std::hash<std::string>{}(log));
        for (auto &&[type, part] : parts) {
            h = hashCombine(h, part->hash());
        }
        return h;
    }
};

class Sketch {
    Schedule schedule_;
    Schedule generatedSchedule_;
    std::vector<SketchTarget> targets_;
    int nowTargetNum_{0};
    double time_{0};
    bool scheduleGenerated_{false};
    std::string code_;
    Func lowered_;
    std::vector<double> feature_;

  public:
    Sketch() = default;
    Sketch(const Sketch &) = default;
    Sketch(const Schedule &schedule, std::vector<ForsWithDataReuse> targets)
        : schedule_(schedule.clone()), nowTargetNum_(targets.size() - 1) {
        for (auto &target : targets) {
            targets_.emplace_back(std::move(target));
        }
    }

    [[nodiscard]] Sketch clone() const {
        Sketch ret;
        ret.schedule_ = schedule_.clone();
        ret.targets_ = targets_;
        ret.nowTargetNum_ = nowTargetNum_;
        return ret;
    }

    Sketch genRandAnnotation(std::default_random_engine &gen) const;

    Schedule genSchedule();

    void addPart(const SketchPart &p);
    PartMap &part(int i) { return targets_[i].parts; }

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] std::pair<bool, Sketch>
    genMutation(std::default_random_engine &gen) const;

    [[nodiscard]] std::pair<bool, Sketch>
    genCrossover(const Sketch &sketch, std::default_random_engine &gen) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }

    size_t hash() const;

    SketchTarget &nowTarget() { return targets_[nowTargetNum_]; }
    [[nodiscard]] const SketchTarget &nowTarget() const {
        return targets_[nowTargetNum_];
    }

    void moveToNextTarget() { nowTargetNum_--; }

    int nowTargetNum() const { return nowTargetNum_; }

    void addLog(const std::string &name) {
        targets_[nowTargetNum_].log += name + ";\n";
    }

    Schedule &schedule() { return schedule_; }
    const Schedule &schedule() const { return schedule_; }

    std::string genCode(const Ref<Target> &target);

    const std::string &code() const { return code_; }
    Func lowered() const { return lowered_; }

    std::vector<double> &genFeature();
    std::vector<double> &feature() { return feature_; }
};

} // namespace freetensor

#endif // FREE_TENSOR_SKETCH_H
