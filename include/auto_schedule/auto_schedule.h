#ifndef IR_AUTO_SCHEDULE_H
#define IR_AUTO_SCHEDULE_H

#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <pybind11/pybind11.h>
#include <random>
#include <schedule.h>
#include <unordered_map>
#include <set>

namespace ir {
namespace py = pybind11;

constexpr int EVOLUTIONARY_SEARCH_POPULATION = 32;
constexpr int EVOLUTIONARY_SEARCH_ITERS = 4;
constexpr double EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.6;
constexpr double EVOLUTIONARY_SEARCH_CROSSOVER_PROB = 0.3;

constexpr double INIT_RAND_RATIO = 0.8;

class AutoSchedule {
    Schedule original_;
    Ref<Target> target_;
    Device device_;
    size_t measured_size_;
    std::vector<Sketch> baseSketches_;
    std::vector<Ref<Array>> args_;
    std::unordered_map<std::string, Ref<Array>> kws_;
    bool paramsSet_;
    std::vector<Sketch> measured_sketches_;
    std::set<size_t> measured_hashes_;
    double mn_;
    std::mt19937 rand_gen;
    py::function predict_func;
    py::function update_func;

  private:
    std::vector<double> measure(const std::vector<Schedule> &schedules);

  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device, int measured_size, py::function predict_func, py::function update_func);

    size_t measured_size() const { return measured_size_; }

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws);

    std::vector<Sketch> SearchOneRound(size_t n);

    std::vector<Sketch> EvolutionarySearch(std::vector<Sketch> init,
                                           size_t out_size);

    std::vector<Sketch> GetInitPopulation(size_t n);

    std::vector<Schedule> genSchedules(const std::vector<Sketch> &sketches);

    py::list genFeatures(const std::vector<Schedule> &schedules);

    std::vector<double> get_prediction(const std::vector<Sketch>&);

    std::vector<double> testAndAdd(const std::vector<Sketch> &sketches);

    Schedule getBestSchedule();
};

} // namespace ir

#endif // IR_AUTO_SCHEDULE_H
