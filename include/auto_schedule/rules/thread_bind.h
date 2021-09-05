#ifndef IR_THREAD_BIND_H
#define IR_THREAD_BIND_H

#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <driver/device.h>

#include <array>

namespace ir {
class ThreadBindRule : public Rule {
    std::vector<std::pair<std::string, int>> target;
    Ref<Target> target_;

  public:
    explicit ThreadBindRule(Ref<Target> argTarget_) : target_(argTarget_) {}
    int analyze(Schedule &schedule) override;
    SketchPart genPart(int p) override;
};

class ThreadBindPart : public SketchPartNode {
    static const int THREAD_MAX_NUM = 1024;
    std::vector<std::pair<std::string, int>> targetFor_;
    Ref<Target> target_;

  public:
    void genRandAnnotation() override;
    explicit ThreadBindPart(std::vector<std::pair<std::string, int>>,
                            Ref<Target>);
    void apply(Schedule &schedule) override;
    SketchPart mutate() override;
    SketchPart crossover(const SketchPart &part) override;
    [[nodiscard]] std::vector<int> getAnnotation() const override;

  private:
    void applyOnCPU(Schedule &schedule);
    void applyOnGPU(Schedule &schedule);
};

} // namespace ir

#endif // IR_THREAD_BIND_H
