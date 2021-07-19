#ifndef IR_SKETCH_H
#define IR_SKETCH_H

#include <schedule.h>
#include <vector>

namespace ir {
class SketchPartNode;
typedef Ref<SketchPartNode> SketchPart;
class SketchPartNode {
  public:
    virtual void gen_rand_annotation() = 0;
    virtual SketchPart mutate() { return nullptr; }
    virtual SketchPart crossover(const SketchPart &part) { return nullptr; };
    virtual void apply(Schedule &schedule) = 0;
    virtual std::vector<int> get_annotation() const = 0;
    virtual ~SketchPartNode() = default;
};

class Sketch {
    Schedule schedule_;
    std::vector<SketchPart> parts_;
    bool annotated;

  public:
    Sketch() = default;
    explicit Sketch(const Schedule &schedule);
    Sketch gen_rand_annotation() const;
    Schedule gen_schedule() const;
    void add_part(const SketchPart &);
    bool operator<(const Sketch &a) const;
    std::vector<int> get_annotation() const;
    [[nodiscard]] std::pair<bool, Sketch> gen_mutation() const;
    [[nodiscard]] std::pair<bool, Sketch>
    gen_crossover(const Sketch &sketch) const;
    double time;
};
} // namespace ir
#endif // IR_SKETCH_H
