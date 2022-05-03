#ifndef FREE_TENSOR_AUTO_SCHEDULE_H
#define FREE_TENSOR_AUTO_SCHEDULE_H

#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <pybind11/pybind11.h>
#include <random>
#include <schedule.h>
#include <set>
#include <unordered_map>

namespace freetensor {
namespace py = pybind11;

constexpr int EVOLUTIONARY_SEARCH_POPULATION = 128;
constexpr int EVOLUTIONARY_SEARCH_ITERS = 4;
constexpr double EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.6;
constexpr double EVOLUTIONARY_SEARCH_CROSSOVER_PROB = 0.3;

constexpr double INIT_RAND_RATIO = 0.7;

class AutoSchedule {
    Schedule original_;
    Ref<Target> target_;
    Device device_;
    size_t measuredSize_;
    std::vector<Ref<Sketch>> baseSketches_;
    std::vector<Ref<Array>> args_;
    std::unordered_map<std::string, Ref<Array>> kws_;
    bool paramsSet_;
    std::vector<Ref<Sketch>> measuredSketches_;
    std::set<size_t> measuredHashes_;
    double mn_;
    std::default_random_engine randGen_;
    py::function predictFunc_;
    py::function updateFunc_;
    std::vector<Ref<Rule>> rules_;

  private:
    std::vector<double> measure(std::vector<Ref<Sketch>> &sketches);

  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Device &device, int measuredSize,
                 py::function predictFunc, py::function updateFunc);

    size_t measuredSize() const { return measuredSize_; }

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws);

    void searchOneRound(size_t n);

    std::vector<Ref<Sketch>> evolutionarySearch(std::vector<Ref<Sketch>> init,
                                                size_t outSize);

    std::vector<Ref<Sketch>> getInitPopulation(size_t n);

    std::vector<Ref<Sketch>> getRandPopulation(size_t nRand);

    py::list genFeatures(std::vector<Ref<Sketch>> &sketches);

    std::vector<double> getPrediction(std::vector<Ref<Sketch>> &sketches_in);

    std::vector<double> testAndAdd(std::vector<Ref<Sketch>> &sketches_in);

    Schedule getBestSchedule();

    void genSketches();
    Sketch getInitSketch();
    Stmt testCacheWrite();
    Schedule testMultiLevelTilingWithFusion(int nLevel);
    Schedule testThreadBind();
    Schedule testCacheRead();
};

} // namespace freetensor

#endif // FREE_TENSOR_AUTO_SCHEDULE_H
