#include <cmath>
#include <queue>
#include <utility>

#include <analyze/find_elementwise.h>
#include <analyze/structural_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/cache_write.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/rules/parallelize.h>
#include <auto_schedule/rules/skip.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/rules/unroll.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen.h>
#include <container_utils.h>
#include <driver.h>
#include <lower.h>
#include <omp_utils.h>

namespace freetensor {

static size_t decideSeed(std::optional<size_t> seed, int verbose) {
    size_t ret;
    if (seed.has_value()) {
        ret = *seed;
    } else {
        ret = std::random_device{}();
    }
    if (verbose >= 1) {
        logger() << "Random seed: " << ret << std::endl;
    }
    return ret;
}

AutoSchedule::AutoSchedule(
    const Schedule &schedule, const Ref<Target> &target,
    const Ref<Device> &device,
    const std::function<Predicts(const Features &)> &predictFunc,
    const std::function<void(const Features &, const Predicts &)> &updateFunc,
    std::string tag, int minBlockSize, std::optional<size_t> randomSeed,
    const std::optional<std::unordered_set<std::string>> &ruleSet, int verbose)
    : original_(schedule.fork()), target_(target), device_(device),
      paramsSet_(false), rng_(decideSeed(randomSeed, verbose)),
      predictFunc_(std::move(predictFunc)), updateFunc_(std::move(updateFunc)),
      tag_(std::move(tag)), minBlockSize_(minBlockSize), verbose_(verbose) {
    flop_ = 0;
    auto opCnt =
        structuralFeature(original_.ast())[original_.ast()->id()].opCnt_;
    for (auto cnt : opCnt) {
        flop_ += cnt.second;
    }

#define ADD_RULE(name, RuleType, ...)                                          \
    if (!ruleSet || ruleSet->count(name)) {                                    \
        rules_.emplace_back(name, Ref<RuleType>::make(__VA_ARGS__));           \
    }

    if (target->type() == TargetType::CPU) {
        ADD_RULE("cache_write", CacheWriteRule, target->type(), verbose_);
        ADD_RULE("multi_level_tiling_with_fusion",
                 MultiLevelTilingWithFusionRule, target->type());
        ADD_RULE("multi_level_tiling", MultiLevelTilingRule, target->type());
        ADD_RULE("parallelize", ParallelizeRule);
        ADD_RULE("unroll", UnrollRule, target->type());
    } else {
        ADD_RULE("cache_write", CacheWriteRule, target->type(), verbose_);
        ADD_RULE("multi_level_tiling_with_fusion",
                 MultiLevelTilingWithFusionRule, target->type(), minBlockSize);
        ADD_RULE("thread_bind", ThreadBindRule);
        ADD_RULE("unroll", UnrollRule, target->type());
    }

#undef ADD_RULE

    rules_.emplace_back("skip", Ref<SkipRule>::make());
}

void AutoSchedule::setParams(
    const std::vector<Ref<Array>> &args,
    const std::unordered_map<std::string, Ref<Array>> &kws) {
    args_ = args;
    kws_ = kws;
    paramsSet_ = true;
}

std::pair<std::vector<double>, std::vector<double>>
AutoSchedule::measure(const std::vector<Ref<Sketch>> &sketches) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    if (verbose_ >= 1) {
        logger() << "Compiling code" << std::endl;
    }
    size_t n = sketches.size();
    std::vector<Ref<Driver>> drivers(n);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < n; i++) {
        try {
            auto lowered = sketches[i]->lowered();
            auto code = codeGen(lowered, target_);
            drivers[i] = Ref<Driver>::make(lowered, code, device_);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            drivers[i] = nullptr;
        }
    }

    if (verbose_ >= 1) {
        logger() << "Measuring time" << std::endl;
    }
    std::vector<double> times, stddevs;
    times.reserve(n);
    stddevs.reserve(n);
    for (size_t i = 0; i < n; i++) {
        ASSERT(paramsSet_);
        try {
            if (!drivers[i].isValid()) {
                times.emplace_back(INFINITY);
                stddevs.emplace_back(0);
                continue;
            }
            drivers[i]->setArgs(args_, kws_);
            auto [avg, stddev] = drivers[i]->time(100, 10);
            times.emplace_back(avg);
            stddevs.emplace_back(stddev);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            times.emplace_back(INFINITY);
            stddevs.emplace_back(0);
        }
    }
    return std::make_pair(times, stddevs);
}

void AutoSchedule::searchOneRound(size_t n, size_t nExploit, size_t nExplore) {
    ASSERT(n == nExploit + nExplore);
    if (baseSketches_.empty()) { // first time
        genSketches();
        testAndAdd(getRandPopulation(n));
    } else {
        testAndAdd(evolutionarySearch(nExploit));
        testAndAdd(getRandPopulation(nExplore));
    }
    auto bs = getBestSchedule();
    if (verbose_ >= 1) {
        logger() << "Best schedule:" << std::endl;
        for (auto log : bs.logs().asVector()) {
            logger() << *log << std::endl;
        }
        logger() << "Best AST: " << std::endl
                 << measuredSketches_[0]->genSchedule().ast() << std::endl;
    }
}

std::vector<std::vector<double>>
AutoSchedule::genFeatures(const std::vector<Ref<Sketch>> &sketches) {
    if (verbose_ >= 1) {
        logger() << "Generating features" << std::endl;
    }
    auto n = sketches.size();
    std::vector<std::vector<double>> featureList(n);
    exceptSafeParallelFor<size_t>(
        0, sketches.size(), 1,
        [&](size_t i) { featureList[i] = sketches[i]->feature(); },
        omp_sched_dynamic);
    return featureList;
}

std::vector<double>
AutoSchedule::testAndAdd(const std::vector<Ref<Sketch>> &sketches) {
    auto features = genFeatures(sketches);
    size_t n = sketches.size();
    ASSERT(features.size() == n);
    auto &&[times, stddevs] = measure(sketches);
    std::vector<double> flopsList;
    for (auto [t, stddev] : views::zip(times, stddevs)) {
        if (t < 1e20) {
            flopsList.emplace_back(flop_ / t);
        }
    }
    updateFunc_(features, flopsList);
    double allAvg = 0, maxStddevPercent = 0;
    int cnt = 0;
    for (auto &&[t, stddev, sketch] : views::zip(times, stddevs, sketches)) {
        if (t < 1e20) {
            cnt++;
            allAvg += t;
            maxStddevPercent = std::max(maxStddevPercent, stddev / t);
            measuredSketches_.emplace_back(sketch);
            measuredSketches_.back()->setTime(t);
            measuredHashes_.insert(sketch->hash());
        }
    }
    allAvg /= cnt;
    std::sort(times.begin(), times.end());
    std::sort(measuredSketches_.begin(), measuredSketches_.end(),
              [](const auto &a, const auto &b) { return *a < *b; });
    if (verbose_ >= 1) {
        logger() << "Global min: " << measuredSketches_.front()->time()
                 << std::endl;
        logger() << "This round: min: " << times[0] << " avg: " << allAvg
                 << " mid: " << times[(times.size() - 1) / 2] << std::endl;
        logger() << "Max_sketch(estimated standard deviation of average "
                    "measurements / average measurements) = "
                 << (maxStddevPercent * 100) << "%" << std::endl;
    }
    return times;
}

Schedule AutoSchedule::getBestSchedule() {
    if (measuredSketches_.empty()) {
        return {};
    }
    return measuredSketches_[0]->genSchedule();
}

double AutoSchedule::getBestTime() {
    if (measuredSketches_.empty()) {
        return INFINITY;
    }
    return measuredSketches_[0]->time();
}

std::vector<Ref<Sketch>> AutoSchedule::getRandPopulation(size_t nRand) {
    std::vector<Ref<Sketch>> ret;
    std::set<size_t> used(measuredHashes_);
    int roundUnchanged = 0;
    while (ret.size() < nRand) {
        std::vector<Ref<Sketch>> now(nRand);
        size_t nThisTurn = nRand;
        // Use static schedule for deterministic random numbers
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nThisTurn; i++) {
            now[i] = baseSketches_[randomInt(baseSketches_.size() - 1, rng_)]
                         ->genRandAnnotation(rng_);
        }
        roundUnchanged++;
        for (size_t i = 0; i < nThisTurn; i++) {
            if (!now[i].isValid()) {
                continue;
            }
            size_t h = now[i]->hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
                roundUnchanged = 0;
            }
            if (ret.size() >= nRand) {
                break;
            }
        }
        if (roundUnchanged > 10) {
            break;
        }
    }
    ASSERT(ret.size() == nRand);
    return ret;
}

std::vector<Ref<Sketch>> AutoSchedule::evolutionarySearch(size_t outSize) {
    // Meta-parameters used for evolutionary search. Population in an
    // evolutionary search is different from the global population
    constexpr size_t EVOLUTIONARY_SEARCH_POPULATION = 512;
    constexpr size_t EVOLUTIONARY_SEARCH_INIT_EXPLORE_CNT = 358;
    constexpr size_t EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT = 128;
    constexpr int EVOLUTIONARY_SEARCH_ITERS = 4;
    constexpr double EVOLUTIONARY_SEARCH_MUTATION_PROB = 0.6;
    constexpr double EVOLUTIONARY_SEARCH_CROSSOVER_PROB = 0.3;

    if (verbose_ >= 1) {
        logger() << "Evolutionary search" << std::endl;
    }

    // init is not necessarily of full population
    std::vector<Ref<Sketch>> init =
        getRandPopulation(EVOLUTIONARY_SEARCH_INIT_EXPLORE_CNT);
    if (measuredSketches_.size() > EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT) {
        ASSERT(EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT > 0);
        measuredSketches_.resize(EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT);
    }
    for (size_t i = 0; i < EVOLUTIONARY_SEARCH_INIT_EXPLOIT_CNT &&
                       i < measuredSketches_.size();
         i++) {
        init.emplace_back(measuredSketches_[i]);
    }

    std::vector<Ref<Sketch>> v1 = std::move(init), v2;
    typedef std::pair<Ref<Sketch>, double> SketchPred;
    std::vector<SketchPred> heap;
    std::set<size_t> heapHashes(measuredHashes_);
    auto cmp = [](const SketchPred &a, const SketchPred &b) {
        return a.second > b.second;
    };
    for (int i = 0; i <= EVOLUTIONARY_SEARCH_ITERS; i++) {
        if (verbose_ >= 1) {
            logger() << "search round " << i << std::endl;
        }
        auto pred = getPrediction(v1);
        for (size_t j = 0; j < v1.size(); j++) {
            size_t hash = v1[j]->hash();
            auto flops = pred[j];
            if (!heapHashes.count(hash)) {
                if (heap.size() < outSize) {
                    heapHashes.insert(hash);
                    heap.emplace_back(v1[j], flops);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (flops > heap[0].second) {
                    heapHashes.erase(heap[0].first->hash());
                    heapHashes.insert(hash);
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = std::make_pair(v1[j], flops);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }
        if (i == EVOLUTIONARY_SEARCH_ITERS) {
            break;
        }

        auto prob =
            getProbFromPredict(pred); // Sketches with higher performance are
                                      // picked with higher probabilities
        std::discrete_distribution<int> probDist(prob.begin(), prob.end());

        v2.resize(EVOLUTIONARY_SEARCH_POPULATION);
        // Use static schedule for deterministic random numbers
#pragma omp parallel for schedule(static)
        for (size_t j = 0; j < EVOLUTIONARY_SEARCH_POPULATION; j++) {
            std::discrete_distribution<int> actionDist(
                {EVOLUTIONARY_SEARCH_MUTATION_PROB,
                 EVOLUTIONARY_SEARCH_CROSSOVER_PROB,
                 1 - EVOLUTIONARY_SEARCH_MUTATION_PROB -
                     EVOLUTIONARY_SEARCH_CROSSOVER_PROB});
            switch (actionDist(rng_)) {
            case 0: // Mutation
                while (true) {
                    int a = probDist(rng_);
                    if (auto mutated = v1[a]->genMutation(rng_);
                        mutated.isValid()) {
                        v2[j] = mutated;
                        break;
                    }
                }
                break;
            case 1: // Crossover
                while (true) {
                    int a = probDist(rng_), b = probDist(rng_);
                    if (a != b) {
                        if (auto crossed = v1[a]->genCrossover(*v1[b], rng_);
                            crossed.isValid()) {
                            v2[j] = crossed;
                            break;
                        }
                    }
                }
                break;
            case 2: { // Direct inheritance
                int a = probDist(rng_);
                ASSERT(v1[a].isValid());
                v2[j] = v1[a];
                break;
            }
            default:
                ASSERT(false);
            }
        }

        std::swap(v1, v2);
    }
    std::sort(heap.begin(), heap.end(), cmp);
    std::vector<Ref<Sketch>> ret;
    for (auto &i : heap) {
        ret.push_back(std::move(i.first));
    }
    return ret;
}

std::vector<double>
AutoSchedule::getPrediction(std::vector<Ref<Sketch>> &sketches_in) {
    std::vector<int> index;
    std::vector<double> ret(sketches_in.size());
    index.reserve(sketches_in.size());
    std::vector<Ref<Sketch>> sketches;
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < sketches_in.size(); i++) {
        sketches_in[i]->genSchedule();
    }
    for (size_t i = 0; i < sketches_in.size(); i++) {
        if (sketches_in[i]->schedule().ast().isValid()) {
            index.push_back(i);
            sketches.push_back(sketches_in[i]);
        } else {
            ret[i] = -INFINITY;
        }
    }
    auto featureList = genFeatures(sketches);
    if (verbose_ >= 1) {
        logger() << "Getting predictions" << std::endl;
    }
    auto predList = predictFunc_(featureList);
    for (size_t i = 0; i < predList.size(); i++) {
        ret[index[i]] = predList[i];
    }
    return ret;
}

void AutoSchedule::genSketches() {
    auto subs = findMultiLevelTiling(original_.ast());
    if (subs.empty()) {
        return;
    }
    auto initSketch = Ref<Sketch>::make(target_, original_, subs);
    std::queue<Ref<Sketch>> q;
    q.push(std::move(initSketch));
    while (!q.empty()) {
        auto nowSketch = std::move(q.front());
        q.pop();
        for (auto &[name, rule] : rules_) {
            if (auto status = rule->analyze(*nowSketch);
                status != RuleStatus::Skip) {
                auto sketches = rule->genPart(*nowSketch);
                for (auto &sketch : sketches) {
                    if (sketch->nowSubNum() == -1) {
                        baseSketches_.emplace_back(sketch);
                    } else {
                        q.push(std::move(sketch));
                    }
                }
                if (status == RuleStatus::ApplyAndSkipRest) {
                    break;
                }
            }
        }
    }
}

Schedule
AutoSchedule::testRound(const std::unordered_map<std::string, int> &nthSketch) {
    auto subs = findMultiLevelTiling(original_.ast());
    ASSERT(!subs.empty());
    auto sketch = Ref<Sketch>::make(target_, original_, subs);
    for (auto &[name, rule] : rules_) {
        if (auto status = rule->analyze(*sketch); status != RuleStatus::Skip) {
            auto newSketches = rule->genPart(*sketch);
            if (nthSketch.count(name)) {
                sketch = std::move(newSketches.at(nthSketch.at(name)));
            } else {
                ASSERT(!newSketches.empty());
                sketch = std::move(newSketches.front());
            }
        }
    }
    for (auto &&[type, part] : sketch->part(0)) {
        part->genFakeAnnotation(rng_);
    }
    return sketch->genSchedule();
}

} // namespace freetensor
