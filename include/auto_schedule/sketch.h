#ifndef IR_SKETCH_H
#define IR_SKETCH_H

#include <random>
#include <schedule.h>
#include <vector>

namespace ir {

class SketchPartNode;

typedef Ref<SketchPartNode> SketchPart;

enum class SketchPartType : int {
    MultiLevelTiling,
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
    std::vector<SketchPart> parts_;
    double time_;

  public:
    Sketch() = default;

    Sketch genRandAnnotation(std::default_random_engine &gen) const;

    Schedule genSchedule(const Schedule &original) const;

    void addPart(const SketchPart &);

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] std::pair<bool, Sketch>
    genMutation(std::default_random_engine &gen) const;

    [[nodiscard]] std::pair<bool, Sketch>
    genCrossover(const Sketch &sketch, std::default_random_engine &gen) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }

    size_t hash() const;
};

} // namespace ir

#endif // IR_SKETCH_H
