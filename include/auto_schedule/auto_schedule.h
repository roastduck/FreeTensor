#ifndef IR_AUTO_SCHEDULE_H
#define IR_AUTO_SCHEDULE_H

#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <schedule.h>
#include <unordered_map>
#include <auto_schedule/sketch.h>

namespace ir {

class AutoSchedule {
  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device)
        : schedule_(schedule), target(target), device(device), params_set(false) {}

    void set_params(const std::vector<Array *> &args,
                    const std::unordered_map<std::string, Array *> &kws);

    Schedule run(unsigned int round = 5, unsigned int save = 20,
             unsigned int extend = 80);

    double measure(const Schedule &schedule);

    double measure(Sketch &sketch);

  private:
    Schedule schedule_;
    Ref<Target> target;
    Device device;
    std::vector<Array *> args_;
    std::unordered_map<std::string, Array *> kws_;
    bool params_set;
    std::vector<Sketch> candidates;
    int n_candidates;
};

} //namespace ir
#endif // IR_AUTO_SCHEDULE_H
