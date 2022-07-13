#ifndef FREE_TENSOR_SKETCH_H
#define FREE_TENSOR_SKETCH_H

#include <map>
#include <utility>
#include <vector>

#include <analyze/find_multi_level_tiling.h>
#include <opt.h>
#include <random.h>
#include <schedule.h>

namespace freetensor {

class SketchPartNode;

typedef Ref<SketchPartNode> SketchPart;

enum class SketchPartType : int {
    MultiLevelTiling = 0,
    MultiLevelTilingWithFusion = 1,
    ThreadBind = 2,
    Unroll = 3,
    Parallelize = 4,
};

struct SubSketch;

/**
 * How a sub-program is scheduled by a `Rule`
 */
class SketchPartNode {
  protected:
    typedef OpenMPRandomEngine RNG;

  public:
    virtual ~SketchPartNode() = default;

    /**
     * Randomly annotate the part
     */
    virtual void genRandAnnotation(RNG &gen) = 0;

    /**
     * Fake annotation only used for testing
     *
     * Defaults to a random annotation
     */
    virtual void genFakeAnnotation(RNG &gen) { genRandAnnotation(gen); }

    /**
     * Randomly mutate the part
     */
    virtual bool mutate(RNG &gen) { return false; }

    /**
     * Crossbreed with another part
     */
    virtual bool crossover(const SketchPart &part, RNG &gen) { return false; };

    /**
     * Make actual transformations on a Schedule, according to the annotation
     */
    virtual void apply(Schedule &schedule, SubSketch &subSketch) = 0;

    virtual SketchPartType partType() = 0;
    virtual std::vector<int> getAnnotation() const = 0;
    virtual size_t hash() const = 0;
    virtual SketchPart clone() const = 0;
};

typedef std::map<SketchPartType, SketchPart> PartMap;

/**
 * How a sub-program is scheduled
 *
 * A sub-program is scheduled by several `Rule`s. `Rule`s are applied to the
 * sub-program one by one. In each `Rule`, the sub-program is scheduled
 * according to the information saved in a `SketchPart`
 */
struct SubSketch {
    ForsWithDataReuse target;
    PartMap parts;
    std::string log;
    explicit SubSketch(ForsWithDataReuse target) : target(std::move(target)) {}
    [[nodiscard]] bool canCrossOver(const SubSketch &other) const {
        return log == other.log;
    }

    SubSketch(const SubSketch &other) : target(other.target), log(other.log) {
        for (auto &&[type, part] : other.parts) {
            parts.emplace(type, part->clone());
        }
    }

    [[nodiscard]] bool hasPart(SketchPartType tp) const {
        return parts.count(tp);
    }

    [[nodiscard]] SketchPart getPart(SketchPartType tp) const {
        if (hasPart(tp)) {
            return parts.at(tp);
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

/**
 * How a `Func` is scheduled
 *
 * A `Func` is decomposed as several sub-programs, each is scheduled according
 * to the information saved in a `SubSketch`
 */
class Sketch {
    typedef OpenMPRandomEngine RNG;

    Ref<Target> target_;
    std::vector<SubSketch> subs_;
    int nowSubNum_{0};
    double time_{0};

    Schedule schedule_; // Original schedule (before genSchedule)

    // Cached schedule, lower and feature result. Data flow:
    // schedule -> lowered -> feature ----+-> upd model
    //                |                   |
    //                +-----> code ----> time
    Opt<Schedule> genSchedule_;
    Func lowered_;
    Opt<std::vector<double>> feature_;

  public:
    Sketch() = default;
    Sketch(const Sketch &) = default;
    Sketch(const Ref<Target> &target, const Schedule &schedule,
           std::vector<ForsWithDataReuse> subs)
        : target_(target), nowSubNum_(subs.size() - 1),
          schedule_(schedule.clone()) {
        for (auto &sub : subs) {
            subs_.emplace_back(std::move(sub));
        }
    }

    [[nodiscard]] Ref<Sketch> clone() const {
        auto ret = Ref<Sketch>::make();
        ret->target_ = target_;
        ret->schedule_ = schedule_.clone();
        ret->subs_ = subs_;
        ret->nowSubNum_ = nowSubNum_;
        return ret;
    }

    Ref<Sketch> genRandAnnotation(RNG &gen) const;

    void addPart(const SketchPart &p);
    PartMap &part(int i) { return subs_[i].parts; }

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] Ref<Sketch> genMutation(RNG &gen) const;

    [[nodiscard]] Ref<Sketch> genCrossover(const Sketch &sketch,
                                           RNG &gen) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }

    size_t hash() const;

    SubSketch &nowSubSketch() { return subs_[nowSubNum_]; }
    [[nodiscard]] const SubSketch &nowSubSketch() const {
        return subs_[nowSubNum_];
    }

    void moveToNextSub() { nowSubNum_--; }

    int nowSubNum() const { return nowSubNum_; }

    void addLog(const std::string &name) {
        subs_[nowSubNum_].log += name + ";\n";
    }

    /**
     * Get original (before genSchedule) schedule
     *
     * Rules can apply some preprocessing schedules during the `genPart` stage.
     * To do this, they can directly modify this "original" schedule.
     *
     * Later in the `genSchedule` staged, the "original" schedule will be copied
     * and further modified by tuning.
     */
    Schedule &schedule() { return schedule_; }
    const Schedule &schedule() const { return schedule_; }

    /**
     * Generate a Schedule. The result is cached
     */
    const Schedule &genSchedule();

    /**
     * Lower a generated Schedule to an AST. The result is cached
     */
    const Func &lowered();

    /**
     * Generate a feature from a lowered AST. The result is cached
     */
    const std::vector<double> &feature();
};

} // namespace freetensor

#endif // FREE_TENSOR_SKETCH_H
