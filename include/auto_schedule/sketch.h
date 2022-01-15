#ifndef IR_SKETCH_H
#define IR_SKETCH_H

#include <schedule.h>
#include <vector>
#include <random>

namespace ir {

class SketchPartNode;

typedef Ref<SketchPartNode> SketchPart;

class SketchPartNode {
  public:
    virtual void genRandAnnotation(std::mt19937 gen) = 0;
    virtual SketchPart mutate(std::mt19937 &gen) { return nullptr; }
    virtual SketchPart crossover(const SketchPart &part, std::mt19937 &gen) { return nullptr; };
    virtual void apply(Schedule &schedule) = 0;
    virtual std::vector<int> getAnnotation() const = 0;
    virtual ~SketchPartNode() = default;
    virtual size_t hash() const = 0;
};

class Sketch {
    std::vector<SketchPart> parts_;
    double time_;

  public:
    Sketch() = default;

    Sketch genRandAnnotation(std::mt19937 gen) const;

    Schedule genSchedule(const Schedule &original) const;

    void addPart(const SketchPart &);

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] std::pair<bool, Sketch> genMutation() const;

    [[nodiscard]] std::pair<bool, Sketch>
    genCrossover(const Sketch &sketch) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }

    size_t hash() const;
};

} // namespace ir

#endif // IR_SKETCH_H
