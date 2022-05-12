#ifndef FREE_TENSOR_AUTO_SCHEDULE_H
#define FREE_TENSOR_AUTO_SCHEDULE_H

#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <functional>
#include <random>
#include <schedule.h>
#include <set>
#include <unordered_map>

namespace freetensor {

constexpr int EVOLUTIONARY_SEARCH_POPULATION = 128;
constexpr int EVOLUTIONARY_SEARCH_ITERS = 4;
constexpr double EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.6;
constexpr double EVOLUTIONARY_SEARCH_CROSSOVER_PROB = 0.3;

constexpr double INIT_RAND_RATIO = 0.7;

class AutoSchedule {
  public:
    typedef std::vector<std::vector<double>> Features;
    typedef std::vector<double> Predicts;

  private:
    Schedule original_;
    Ref<Target> target_;
    Ref<Device> device_;
    size_t measuredSize_;
    std::vector<Ref<Sketch>> baseSketches_;
    std::vector<Ref<Array>> args_;
    std::unordered_map<std::string, Ref<Array>> kws_;
    bool paramsSet_;
    std::vector<Ref<Sketch>> measuredSketches_;
    std::set<size_t> measuredHashes_;
    std::default_random_engine randGen_;
    const std::function<Predicts(const Features &)> &predictFunc_;
    const std::function<void(const Features &, const Predicts &)> &updateFunc_;
    std::vector<Ref<Rule>> rules_;
    double flop_;
    std::string tag_;

  private:
    std::vector<double> measure(std::vector<Ref<Sketch>> &sketches);

  public:
    AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                 const Ref<Device> &device, int measuredSize,
                 const std::function<Predicts(const Features &)> &predictFunc,
                 const std::function<void(const Features &, const Predicts &)>
                     &updateFunc,
                 std::string tag = "");

    size_t measuredSize() const { return measuredSize_; }

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws);

    void searchOneRound(size_t n);

    std::vector<Ref<Sketch>> evolutionarySearch(std::vector<Ref<Sketch>> init,
                                                size_t outSize);

    std::vector<Ref<Sketch>> getInitPopulation(size_t n);

    std::vector<Ref<Sketch>> getRandPopulation(size_t nRand);

    std::vector<std::vector<double>>
    genFeatures(std::vector<Ref<Sketch>> &sketches);

    std::vector<double> getPrediction(std::vector<Ref<Sketch>> &sketches_in);

    std::vector<double> testAndAdd(std::vector<Ref<Sketch>> &sketches_in);

    Schedule getBestSchedule();
    double getBestTime();

    double getFlop() { return flop_; }
    std::string getTag() { return tag_; }

    void genSketches();
    Sketch getInitSketch();
    Stmt testCacheWrite();
    Schedule testMultiLevelTilingWithFusion(int nLevel);
    Schedule testThreadBind();
    Schedule testCacheRead();
    Schedule testUnroll();
};

} // namespace freetensor

#endif // FREE_TENSOR_AUTO_SCHEDULE_H
