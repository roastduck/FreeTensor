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
        : schedule_(schedule), target(target), device(device),
          params_set(false) {}

    void set_params(const std::vector<Array *> &args,
                    const std::unordered_map<std::string, Array *> &kws);

    std::pair<std::vector<std::vector<int>>, std::vector<double>>
    init(int _n_candidates);

    std::vector<Sketch> get_random_sketches(size_t n);
    std::pair<std::vector<std::vector<int>>, std::vector<double>>
    test_and_add(const std::vector<Sketch> &sketches);
    Schedule get_best_schedule();
    //    Schedule run(unsigned int round = 5, unsigned int save = 20,
    //             unsigned int extend = 80);

    double measure(const Schedule &schedule);

    double measure(const Sketch &sketch);

  private:
    Schedule schedule_;
    Ref<Target> target;
    Device device;
    std::vector<Array *> args_;
    std::unordered_map<std::string, Array *> kws_;
    bool params_set;
    std::vector<Sketch> candidates;
    size_t n_candidates;
    double mn_;
};

} // namespace ir
#endif // IR_AUTO_SCHEDULE_H
