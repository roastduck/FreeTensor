#ifndef FREE_TENSOR_AUTO_SCHEDULE_H
#define FREE_TENSOR_AUTO_SCHEDULE_H

#include <functional>
#include <random>
#include <schedule.h>
#include <set>
#include <unordered_map>

#include <auto_schedule/rule.h>
#include <auto_schedule/sketch.h>
#include <driver/array.h>
#include <driver/device.h>
#include <driver/target.h>
#include <random.h>

namespace freetensor {

class AutoSchedule {
  public:
    typedef std::vector<std::vector<double>> Features;
    typedef std::vector<double> Predicts;

  private:
    Schedule original_;
    Ref<Target> target_;
    Ref<Device> device_;
    std::vector<Ref<Sketch>> baseSketches_;
    std::vector<Ref<Array>> args_;
    std::unordered_map<std::string, Ref<Array>> kws_;
    bool paramsSet_;
    std::vector<Ref<Sketch>> measuredSketches_; // sorted from fast to slow
    std::set<size_t> measuredHashes_;
    OpenMPRandomEngine rng_;
    std::function<Predicts(const Features &)> predictFunc_;
    std::function<void(const Features &, const Predicts &)> updateFunc_;
    std::vector<std::pair<std::string, Ref<Rule>>> rules_; // (name, rule)
    double flop_;
    std::string tag_;
    int minBlockSize_{0};
    std::optional<std::unordered_set<std::string>> ruleSet_;
    int verbose_ = 0;
    std::function<std::pair<std::vector<double>, std::vector<double>>(
        int rounds, int warmups,
        const std::tuple<const Ref<Target> &, const Ref<Device> &,
                         const std::vector<Ref<Array>> &,
                         const std::unordered_map<std::string, Ref<Array>> &>
            &att,
        const std::vector<Func> &funcs)>
        remoteMeasureSubmit_;

  private:
    /**
     * Compile and measure all the sketches
     *
     * @return : list of average time, list of standard deviation
     */
    std::pair<std::vector<double>, std::vector<double>>
    measure(const std::vector<Ref<Sketch>> &sketches);

  public:
    AutoSchedule(
        const Schedule &schedule, const Ref<Target> &target,
        const Ref<Device> &device,
        const std::function<Predicts(const Features &)> &predictFunc,
        const std::function<void(const Features &, const Predicts &)>
            &updateFunc,
        std::string tag = "", int minBlockSize = 0,
        std::optional<size_t> randomSeed = std::nullopt,
        const std::optional<std::unordered_set<std::string>> &ruleSet =
            std::nullopt,
        int verbose = 0,
        const std::function<std::pair<std::vector<double>, std::vector<double>>(
            int rounds, int warmups,
            const std::tuple<
                const Ref<Target> &, const Ref<Device> &,
                const std::vector<Ref<Array>> &,
                const std::unordered_map<std::string, Ref<Array>> &> &att,
            const std::vector<Func> &funcs)> &remoteMeasureSubmit = {});

    void setParams(const std::vector<Ref<Array>> &args,
                   const std::unordered_map<std::string, Ref<Array>> &kws);

    void searchOneRound(size_t n, size_t nExploit, size_t nExplore);

    std::vector<Ref<Sketch>> evolutionarySearch(size_t outSize);

    std::vector<Ref<Sketch>> getRandPopulation(size_t nRand);

    std::vector<std::vector<double>>
    genFeatures(const std::vector<Ref<Sketch>> &sketches);

    std::vector<double> getPrediction(std::vector<Ref<Sketch>> &sketches_in);

    std::vector<double> testAndAdd(const std::vector<Ref<Sketch>> &sketches_in);

    Schedule getBestSchedule();
    double getBestTime();

    double getFlop() { return flop_; }
    std::string getTag() { return tag_; }

    void genSketches();

    /**
     * Exercise on fake annotations, for test only
     *
     * @param nthSketch : A map from rule name to integer: which sketch to pick
     */
    Schedule
    testRound(const std::unordered_map<std::string, int> &nthSketch = {});
};

std::pair<std::vector<double>, std::vector<double>> rpcMeasure(
    int rounds, int warmups,
    const std::tuple<const Ref<Target> &, const Ref<Device> &,
                     const std::vector<Ref<Array>> &,
                     const std::unordered_map<std::string, Ref<Array>> &> &att,
    const std::vector<Func> &funcs);

} // namespace freetensor

#endif // FREE_TENSOR_AUTO_SCHEDULE_H
