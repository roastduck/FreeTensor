#ifndef IR_AUTO_SCHEDULE_H
#define IR_AUTO_SCHEDULE_H

#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <memory>
#include <schedule.h>
#include <unordered_map>

namespace ir {

class AutoSchedule {
    Schedule original_;
    Schedule current_;
    Ref<Target> target_;
    Device device_;
    size_t nCandidates_, nPredict_;
    Sketch baseSketch_;
    std::vector<Array *> args_;
    std::unordered_map<std::string, Array *> kws_;
    bool paramsSet_;
    std::vector<Sketch> candidates_;
    std::vector<std::shared_ptr<Rule>> rules_;
    double mn_;

  private:
    std::vector<double> measure(const std::vector<Schedule> &schedules);

  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device, int nCandidates, int nPredict);
    ~AutoSchedule();

    size_t nCandidates() const { return nCandidates_; }
    size_t nPredict() const { return nPredict_; }

    void setParams(const std::vector<Array *> &args,
                   const std::unordered_map<std::string, Array *> &kws);

    std::vector<Sketch> getRandomSketches(size_t n);

    std::vector<Schedule> genSchedules(const std::vector<Sketch> &sketches);

    std::vector<std::vector<double>>
    genFeatures(const std::vector<Schedule> &schedules);

    std::vector<double> testAndAdd(const std::vector<Sketch> &sketches,
                                   const std::vector<Schedule> &schedules);

    Schedule getBestSchedule();

    void addParts();

    int getRulesNum();
    int ruleAnalyse(int idx);
    void ruleApply(int idx, std::vector<int> sketchPartIdx);
    std::shared_ptr<Rule> getRule(int idx);
    void setRule(int idx, std::shared_ptr<Rule> rule);
    int getBestTime();
    Schedule getCurrent() { return current_; }
    void setCurrent(Schedule current) { current_ = current; }
    Sketch getBaseSketch() { return baseSketch_; }
    void setBaseSketch(Sketch baseSketch) { baseSketch_ = baseSketch; }
};

} // namespace ir

#endif // IR_AUTO_SCHEDULE_H
