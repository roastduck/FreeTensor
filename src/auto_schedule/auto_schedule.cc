#include <cmath>

#include <analyze/fixed_length_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>
#include <pybind11/numpy.h>

namespace ir {

AutoSchedule::AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                           const Device &device, int measured_size,
                           py::function predict_func, py::function update_func)
    : original_(schedule), target_(target), device_(device),
      measuredSize_(measured_size), paramsSet_(false), mn_(INFINITY),
      predictFunc_(std::move(predict_func)),
      updateFunc_(std::move(update_func)) {
    MultiLevelTilingRule rule;
    int n = rule.analyze(original_);
    std::cout << "Found" << n << std::endl;
    Sketch sketch;
    sketch.addPart(rule.genPart(0));
    baseSketches_.push_back(sketch);
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
            drivers[i] = Ref<Driver>::make(Driver(func, code, device_));
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }

    std::vector<double> times;
    times.reserve(n);
    for (size_t i = 0; i < n; i++) {
        std::cout << "measure " << i << std::endl;
        ASSERT(paramsSet_);
        drivers[i]->setParams(args_, kws_);
        times.emplace_back(drivers[i]->time(5, 20));
    }
    return times;
}

std::vector<Sketch> AutoSchedule::searchOneRound(size_t n) {
    std::cout << "get init population" << std::endl;
    std::vector<Sketch> init = getInitPopulation(n);
    std::cout << "evolutionary search" << std::endl;
    std::vector<Sketch> best = evolutionarySearch(init, n);

    testAndAdd(best);
    std::vector<Sketch> rand = getRandPopulation(n * 0.2);
    testAndAdd(rand);
    return best;
}

std::vector<Schedule>
AutoSchedule::genSchedules(const std::vector<Sketch> &sketches) {
    size_t n = sketches.size();
    std::vector<Schedule> ret(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            ret[i] = sketches[i].genSchedule(original_);
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }
    return ret;
}

py::list AutoSchedule::genFeatures(const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    std::vector<std::vector<double>> feature_vec(n);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            feature_vec[i] = fixedLengthFeature(schedules[i].ast());
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }
    py::list feature_list;
    for (auto &i : feature_vec) {
        py::list feature;
        for (auto j : i) {
            feature.append(py::float_(j));
        }
        feature_list.append(feature);
    }
    return feature_list;
}

std::vector<double>
AutoSchedule::testAndAdd(const std::vector<Sketch> &sketches) {
    std::cout << "schedule" << std::endl;
    auto schedules = genSchedules(sketches);
    std::cout << "feature" << std::endl;
    auto features = genFeatures(schedules);
    size_t n = schedules.size();
    ASSERT(sketches.size() == n);
    std::vector<double> times = measure(schedules);
    py::list times_list;
    for (auto i : times) {
        times_list.append(py::float_(i));
    }
    updateFunc_(features, times_list);
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
    return measuredSketches_[best].genSchedule(original_);
}

std::vector<Sketch> AutoSchedule::getRandPopulation(size_t n_rand) {
    std::vector<Sketch> ret;
    std::set<size_t> used(measuredHashes_);
    std::vector<std::default_random_engine> gens;
    for (size_t i = 0; i < n_rand; i++) {
        gens.push_back(std::default_random_engine((i + i) * randGen_()));
    }
    int iter = 0;
    while (ret.size() < n_rand) {
        std::vector<Sketch> now(n_rand);
#pragma omp parallel for
        for (size_t i = 0; i < n_rand; i++) {
            now[i] = baseSketches_[randomInt(baseSketches_.size() - 1, gens[i])]
                         .genRandAnnotation(gens[i]);
        }
        for (size_t i = 0; i < n_rand; i++) {
            size_t h = now[i].hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
            }
            if (ret.size() >= n_rand) {
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
    size_t n_measured = std::min(n - ret.size(), measuredSketches_.size());
    for (size_t i = 0; i < n_measured; i++) {
        ret.push_back(measuredSketches_[i]);
    }
    return ret;
}

std::vector<Sketch> AutoSchedule::evolutionarySearch(std::vector<Sketch> init,
                                                     size_t out_size) {
    std::vector<Sketch> v1 = std::move(init);
    std::vector<Sketch> v2;
    auto *p1 = &v1;
    auto *p2 = &v2;
    v2.reserve(v1.size());
    typedef std::pair<Sketch, double> SketchPred;
    std::vector<SketchPred> heap;
    std::set<size_t> heap_hashes(measuredHashes_);
    auto cmp = [](const SketchPred &a, const SketchPred &b) {
        return a.second < b.second;
    };

    for (int i = 0; i < EVOLUTIONARY_SEARCH_ITERS; i++) {
        std::cout << "search round " << i << std::endl;
        auto pred = getPrediction(*p1);
        auto prob_sum = getProbSum(pred);
        for (size_t j = 0; j < p1->size(); j++) {
            size_t hash = (*p1)[j].hash();
            auto time = pred[j];
            if (!heap_hashes.count(hash)) {
                if (heap.size() < out_size) {
                    heap_hashes.insert(hash);
                    heap.emplace_back((*p1)[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (time < heap[0].second) {
                    heap_hashes.erase(heap[0].first.hash());
                    heap_hashes.insert(hash);
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = std::make_pair((*p1)[j], time);
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }

        while (p2->size() < EVOLUTIONARY_SEARCH_POPULATION) {
            double r = randomDouble(randGen_);
            if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB) {
                auto nw = (*p1)[randWithProb(prob_sum, randGen_)].genMutation(
                    randGen_);
                if (nw.first) {
                    p2->push_back(nw.second);
                }
            } else if (r < EVOLUTIONARY_SEARCH_MUTATION_PROB +
                               EVOLUTIONARY_SEARCH_CROSSOVER_PROB) {
                int a = randWithProb(prob_sum, randGen_);
                int b = randWithProb(prob_sum, randGen_);
                while (b == a)
                    b = randWithProb(prob_sum, randGen_);
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

std::vector<double>
AutoSchedule::getPrediction(const std::vector<Sketch> &sketches) {
    auto feature_list = genFeatures(genSchedules(sketches));
    py::list pred_list = predictFunc_(feature_list);
    std::vector<double> ret;
    for (auto &i : pred_list) {
        ret.push_back(i.cast<double>());
    }
    return ret;
}

} // namespace ir
