#include <cmath>

#include <analyze/find_elementwise.h>
#include <analyze/fixed_length_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/cache_write.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/rules/multi_level_tiling_with_fusion.h>
#include <auto_schedule/rules/skip.h>
#include <auto_schedule/rules/thread_bind.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>
#include <pybind11/numpy.h>
#include <queue>

namespace ir {

AutoSchedule::AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                           const Device &device, int measuredSize,
                           py::function predictFunc, py::function updateFunc)
    : original_(schedule.clone()), target_(target), device_(device),
      measuredSize_(measuredSize), paramsSet_(false), mn_(INFINITY),
      predictFunc_(std::move(predictFunc)), updateFunc_(std::move(updateFunc)) {
    if (target->type() == TargetType::CPU) {
        rules_.push_back(new CacheWriteRule(target->type()));
        rules_.push_back(new MultiLevelTilingWithFusionRule(target->type()));
        rules_.push_back(new MultiLevelTilingRule(target->type()));
    } else {
        rules_.push_back(new CacheWriteRule(target->type()));
        rules_.push_back(new MultiLevelTilingWithFusionRule(target->type()));
        rules_.push_back(new ThreadBindRule());
    }
    rules_.push_back(new SkipRule());
    std::random_device rd;
    randGen_ = std::default_random_engine(rd());
}

void AutoSchedule::setParams(
    const std::vector<Ref<Array>> &args,
    const std::unordered_map<std::string, Ref<Array>> &kws) {
    args_ = args;
    kws_ = kws;
    paramsSet_ = true;
}

std::vector<double>
AutoSchedule::measure(const std::vector<Schedule> &schedules) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    size_t n = schedules.size();
    std::vector<Ref<Driver>> drivers(n);
    std::cout << "codegen" << std::endl;
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            auto func = lower(schedules[i].func(), target_);
            std::string code;
            if (target_->type() == TargetType::GPU)
                code = codeGenCUDA(func);
            else
                code = codeGenCPU(func);
            drivers[i] = Ref<Driver>::make(func, code, device_);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            exit(-1);
        }
    }

    std::vector<double> times;
    times.reserve(n);
    for (size_t i = 0; i < n; i++) {
        std::cout << "measure " << i << std::endl;
        ASSERT(paramsSet_);
        try {
            drivers[i]->setParams(args_, kws_);
            double t = drivers[i]->time(5, 20);
            if (t < 0.003) {
                t = 1e30;
            }
            times.emplace_back(t);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR measure: " << e.what() << std::endl;
            exit(-1);
        }
    }
    return times;
}

std::vector<Sketch> AutoSchedule::searchOneRound(size_t n) {
    if (baseSketches_.empty()) {
        genSketches();
    }
    std::cout << "get init population" << std::endl;
    std::vector<Sketch> init = getInitPopulation(n);
    std::cout << "evolutionary search" << std::endl;
    std::vector<Sketch> best = evolutionarySearch(init, n);

    testAndAdd(best);
    std::vector<Sketch> rand = getRandPopulation(n * 0.2);
    testAndAdd(rand);
    auto bs = getBestSchedule();
    auto logs = bs.logs();
    for (auto log : logs) {
        std::cout << log << std::endl;
    }
    std::cout << "now best: " << toString(bs.ast()) << std::endl;
    return best;
}

std::vector<Schedule>
AutoSchedule::genSchedules(std::vector<Sketch> &sketches) {
    size_t n = sketches.size();
    std::vector<Schedule> ret(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            ret[i] = sketches[i].genSchedule();
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR schedule: " << e.what() << std::endl;
        }
    }
    return ret;
}

py::list AutoSchedule::genFeatures(const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    std::vector<std::vector<double>> featureVec(n);
//#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            std::cout << schedules[i].ast() << std::endl;
            featureVec[i] = fixedLengthFeature(schedules[i].ast());
            std::cout << "feature got\n";
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR feature: " << e.what() << std::endl;
            exit(-1);
        }
    }
    py::list featureList;
    for (auto &i : featureVec) {
        py::list feature;
        for (auto j : i) {
            feature.append(py::float_(j));
        }
        featureList.append(feature);
    }
    return featureList;
}

std::vector<double> AutoSchedule::testAndAdd(std::vector<Sketch> &sketches_in) {
    std::cout << "schedule" << std::endl;
    auto schedules_in = genSchedules(sketches_in);
    std::vector<Sketch> sketches;
    std::vector<Schedule> schedules;
    for (size_t i = 0; i < sketches_in.size(); i++) {
        if (schedules_in[i].ast().isValid()) {
            sketches.push_back(sketches_in[i]);
            schedules.push_back(schedules_in[i]);
        }
    }
    std::cout << "feature" << std::endl;
    auto features = genFeatures(schedules);
    size_t n = schedules.size();
    ASSERT(sketches.size() == n);
    std::vector<double> times = measure(schedules);
    py::list timesList;
    for (auto i : times) {
        timesList.append(py::float_(i));
    }
    updateFunc_(features, timesList);
    std::make_heap(measuredSketches_.begin(), measuredSketches_.end());
    for (size_t i = 0; i < n; i++) {
        std::cout << "test " << i << std::endl;
        if (measuredSketches_.size() < measuredSize_) {
            measuredSketches_.emplace_back(sketches[i]);
            measuredSketches_.back().setTime(times[i]);
            std::push_heap(measuredSketches_.begin(), measuredSketches_.end());
        } else if (times[i] < measuredSketches_[0].time()) {
            std::pop_heap(measuredSketches_.begin(), measuredSketches_.end());
            measuredSketches_.back() = sketches[i];
            measuredSketches_.back().setTime(times[i]);
            std::push_heap(measuredSketches_.begin(), measuredSketches_.end());
        }
        measuredHashes_.insert(sketches[i].hash());
        mn_ = std::min(times[i], mn_);
    }

    std::cout << "min " << mn_ << " max " << measuredSketches_[0].time()
              << std::endl;
    std::sort(measuredSketches_.begin(), measuredSketches_.end());
    return times;
}

Schedule AutoSchedule::getBestSchedule() {
    int best = 0;
    int time = measuredSketches_[0].time();
    for (size_t i = 0; i < measuredSketches_.size(); i++) {
        if (measuredSketches_[i].time() < time) {
            time = measuredSketches_[i].time();
            best = i;
        }
    }
    return measuredSketches_[best].genSchedule();
}

std::vector<Sketch> AutoSchedule::getRandPopulation(size_t nRand) {
    std::vector<Sketch> ret;
    std::set<size_t> used(measuredHashes_);
    std::vector<std::default_random_engine> gens;
    for (size_t i = 0; i < nRand; i++) {
        gens.push_back(std::default_random_engine((i + i) * randGen_()));
    }
    int iter = 0;
    while (ret.size() < nRand) {
        std::vector<Sketch> now(nRand);
#pragma omp parallel for
        for (size_t i = 0; i < nRand; i++) {
            now[i] = baseSketches_[randomInt(baseSketches_.size() - 1, gens[i])]
                         .genRandAnnotation(gens[i]);
        }
        for (size_t i = 0; i < nRand; i++) {
            size_t h = now[i].hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
            }
            if (ret.size() >= nRand) {
                break;
            }
        }
        if (++iter > 10) {
            break;
        }
    }
    return ret;
}

std::vector<Sketch> AutoSchedule::getInitPopulation(size_t n) {
    std::vector<Sketch> ret = getRandPopulation(n * INIT_RAND_RATIO);
    size_t nMeasured = std::min(n - ret.size(), measuredSketches_.size());
    for (size_t i = 0; i < nMeasured; i++) {
        ret.push_back(measuredSketches_[i]);
    }
    return ret;
}

std::vector<Sketch> AutoSchedule::evolutionarySearch(std::vector<Sketch> init,
                                                     size_t outSize) {
    std::vector<Sketch> v1 = std::move(init);
    std::vector<Sketch> v2;
    auto *p1 = &v1;
    auto *p2 = &v2;
    v2.reserve(v1.size());
    typedef std::pair<Sketch, double> SketchPred;
    std::vector<SketchPred> heap;
    std::set<size_t> heapHashes(measuredHashes_);
    auto cmp = [](const SketchPred &a, const SketchPred &b) {
        return a.second < b.second;
    };

    for (int i = 0; i < EVOLUTIONARY_SEARCH_ITERS; i++) {
        std::cout << "search round " << i << std::endl;
        auto pred = getPrediction(*p1);
        auto probSum = getProbSum(pred);
        for (size_t j = 0; j < p1->size(); j++) {
            size_t hash = (*p1)[j].hash();
            auto time = pred[j];
            if (time > 1e20) {
                continue;
            }
            if (!heapHashes.count(hash)) {
                if (heap.size() < outSize) {
                    heapHashes.insert(hash);
                    heap.emplace_back((*p1)[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (time < heap[0].second) {
                    heapHashes.erase(heap[0].first.hash());
                    heapHashes.insert(hash);
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = std::make_pair((*p1)[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }

        while (p2->size() < EVOLUTIONARY_SEARCH_POPULATION) {
            std::cout << "evo " << p2->size() << std::endl;
            double r = randomDouble(randGen_);
            if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB) {
                auto nw = (*p1)[randWithProb(probSum, randGen_)].genMutation(
                    randGen_);
                if (nw.first) {
                    p2->push_back(nw.second);
                }
            } else if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB +
                               EVOLUTIONARY_SEARCH_CROSSOVER_PROB) {
                int a = randWithProb(probSum, randGen_);
                int b = randWithProb(probSum, randGen_);
                while (b == a)
                    b = randWithProb(probSum, randGen_);
                auto nw = (*p1)[a].genCrossover((*p1)[b], randGen_);
                if (nw.first) {
                    p2->push_back(nw.second);
                }
            } else {
                p2->push_back((*p1)[randomInt(p1->size() - 1, randGen_)]);
            }
        }

        std::swap(p1, p2);
        p2->clear();
    }
    std::sort(heap.begin(), heap.end(), cmp);
    std::vector<Sketch> ret;
    for (auto &i : heap) {
        ret.push_back(std::move(i.first));
    }
    return ret;
}

std::vector<double> AutoSchedule::getPrediction(std::vector<Sketch> &sketches) {
    auto schedules_in = genSchedules(sketches);
    std::cout << "schedule completed" << std::endl;
    std::vector<int> index;
    std::vector<double> ret(schedules_in.size());
    index.reserve(schedules_in.size());
    std::vector<Schedule> schedules;
    for (size_t i = 0; i < schedules_in.size(); i++) {
        if (schedules_in[i].ast().isValid()) {
            index.push_back(i);
            schedules.push_back(schedules_in[i]);
        } else {
            ret[i] = 1e30;
        }
    }
    std::cout << "get prediction" << std::endl;
    auto featureList = genFeatures(schedules);
    std::cout << "got prediction" << std::endl;
    py::list predList = predictFunc_(featureList);
    for (size_t i = 0; i < predList.size(); i++) {
        ret[index[i]] = predList[i].cast<double>();
    }
    return ret;
}
void AutoSchedule::genSketches() {
    auto targets = findMultiLevelTiling(original_.ast());
    if (!targets.size()) {
        return;
    }
    Sketch initSketch(original_, targets);
    std::queue<Sketch> q;
    q.push(std::move(initSketch));
    while (!q.empty()) {
        auto nowSketch = std::move(q.front());
        q.pop();
        for (auto &rule : rules_) {
            if (auto status = rule->analyze(nowSketch);
                status != RuleStatus::Skip) {
                auto sketches = rule->genPart(nowSketch);
                for (auto &sketch : sketches) {
                    if (sketch.nowTargetNum() == -1) {
                        baseSketches_.push_back(sketch);
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

Sketch AutoSchedule::getInitSketch() {
    auto targets = findMultiLevelTiling(original_.ast());
    if (!targets.size()) {
        return {};
    }
    return {original_, targets};
}

Stmt AutoSchedule::testCacheWrite() {
    auto sketch = getInitSketch();
    CacheWriteRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule().ast();
    }
    auto newSketch = rule.genPart(sketch).front();
    return newSketch.schedule().ast();
}

Schedule AutoSchedule::testMultiLevelTilingWithFusion(int nLevel) {
    auto sketch = getInitSketch();
    MultiLevelTilingWithFusionRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule();
    }
    Sketch newSketch = rule.genPart(sketch)[nLevel];
    std::cout << toString(newSketch.schedule().ast()) << std::endl;
    auto part = newSketch.part(0)[SketchPartType::MultiLevelTilingWithFusion]
                    .as<MultiLevelTilingWithFusionPart>();
    part->genSampleAnnotation();
    auto schedule = newSketch.genSchedule();
    return schedule;
}

Schedule AutoSchedule::testThreadBind() {
    auto sketch = getInitSketch();
    MultiLevelTilingWithFusionRule rule(target_->type());
    if (rule.analyze(sketch) == RuleStatus::Skip) {
        return sketch.schedule();
    }
    Sketch newSketch = rule.genPart(sketch)[0];
    std::cout << toString(newSketch.schedule().ast()) << std::endl;
    auto part = newSketch.part(0)[SketchPartType::MultiLevelTilingWithFusion]
                    .as<MultiLevelTilingWithFusionPart>();
    part->genSampleAnnotation();
    newSketch.addPart(Ref<ThreadBindPart>::make());
    auto schedule = newSketch.genSchedule();
    return schedule;
}

} // namespace ir
