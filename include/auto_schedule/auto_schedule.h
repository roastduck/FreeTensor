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
  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device)
        : schedule_(schedule), target_(target), device_(device),
          paramsSet_(false) {}

    void setParams(const std::vector<Array *> &args,
                   const std::unordered_map<std::string, Array *> &kws);

    std::pair<std::vector<std::vector<int>>, std::vector<double>>
    init(int nCandidates);

    std::vector<Sketch> getRandomSketches(size_t n);

    std::pair<std::vector<std::vector<int>>, std::vector<double>>
    testAndAdd(const std::vector<Sketch> &sketches);

    Schedule getBestSchedule();

    std::vector<double> measure(const std::vector<Sketch> &sketches);

  private:
    Schedule schedule_;
    Ref<Target> target_;
    Device device_;
    std::vector<Array *> args_;
    std::unordered_map<std::string, Array *> kws_;
    bool paramsSet_;
    std::vector<Sketch> candidates_;
    size_t nCandidates_;
    double mn_;
};

} // namespace ir
#endif // IR_AUTO_SCHEDULE_H
