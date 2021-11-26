#ifndef IR_AUTO_SCHEDULE_H
#define IR_AUTO_SCHEDULE_H

#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <schedule.h>
#include <unordered_map>

namespace ir {

class AutoSchedule {
    Schedule original_;
    Ref<Target> target_;
    Device device_;
    size_t nCandidates_, nPredict_;
    Sketch baseSketch_;
    std::vector<Ref<Array>> args_;
    std::unordered_map<std::string, Ref<Array>> kws_;
    bool paramsSet_;
    std::vector<Sketch> candidates_;
    double mn_;

  private:
    std::vector<double> measure(const std::vector<Schedule> &schedules);

  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device, int nCandidates, int nPredict);

    size_t nCandidates() const { return nCandidates_; }
    size_t nPredict() const { return nPredict_; }

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws);

    std::vector<Sketch> getRandomSketches(size_t n);

    std::vector<Schedule> genSchedules(const std::vector<Sketch> &sketches);

    std::vector<std::vector<double>>
    genFeatures(const std::vector<Schedule> &schedules);

    std::vector<double> testAndAdd(const std::vector<Sketch> &sketches,
                                   const std::vector<Schedule> &schedules);

    Schedule getBestSchedule();
};

} // namespace ir

#endif // IR_AUTO_SCHEDULE_H
