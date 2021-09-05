#ifndef IR_SKETCH_H
#define IR_SKETCH_H

#include <schedule.h>
#include <vector>

namespace ir {

class SketchPartNode;

typedef Ref<SketchPartNode> SketchPart;

/**
 * A subclass of SketchPartNode must meet the following conditions:
 * 1. The result of apply can only rely on the structure of the AST,
 *    but never the specific parameter in the AST.
 */
class SketchPartNode {
  public:
    virtual void genRandAnnotation() = 0;
    virtual SketchPart mutate() { return nullptr; }
    virtual SketchPart crossover(const SketchPart &part) { return nullptr; };
    virtual void apply(Schedule &schedule) = 0;
    virtual std::vector<int> getAnnotation() const = 0;
    virtual ~SketchPartNode() = default;
};

class Sketch {
    std::vector<SketchPart> parts_;
    double time_;

  public:
    Sketch() = default;

    Sketch genRandAnnotation() const;

    Schedule genSchedule(const Schedule &original) const;

    void addPart(const SketchPart &);

    bool operator<(const Sketch &a) const;

    std::vector<int> getAnnotation() const;

    [[nodiscard]] std::pair<bool, Sketch> genMutation() const;

    [[nodiscard]] std::pair<bool, Sketch>
    genCrossover(const Sketch &sketch) const;

    void setTime(double time) { time_ = time; }
    double time() const { return time_; }
};

} // namespace ir

#endif // IR_SKETCH_H
