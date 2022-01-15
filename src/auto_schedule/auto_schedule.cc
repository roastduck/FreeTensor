#include <cmath>

#include <analyze/fixed_length_feature.h>
#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>

namespace ir {

AutoSchedule::AutoSchedule(const Schedule &schedule, const Ref<Target> &target,
                           const Device &device, int nCandidates, int nPredict)
    : original_(schedule), target_(target), device_(device),
      nCandidates_(nCandidates), nPredict_(nPredict), paramsSet_(false),
      mn_(INFINITY) {
    MultiLevelTilingRule rule;
    int n = rule.analyze(original_);
    std::cout << "Found" << n << std::endl;
    Sketch sketch;
    sketch.addPart(rule.genPart(0));
    baseSketches_.push_back(sketch);
    std::random_device rd;
    rand_gen = std::mt19937(rd());
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
        ASSERT(paramsSet_);
        drivers[i]->setParams(args_, kws_);
        times.emplace_back(drivers[i]->time(5, 20));
    }
    return times;
}

std::vector<Sketch> AutoSchedule::SearchOneRound(size_t n) {
    std::vector<Sketch> ret = GetInitPopulation(n);
//    for (size_t i = 0; i < n; i++) {
//        if (measured_sketches_.size() < nCandidates_) {
//        } else {
//            int mut = random_int(1);
//            if (mut) {
//                auto nw = measured_sketches_[random_int(nCandidates_ - 1)].genMutation();
//                if (nw.first) {
//                    ret.push_back(nw.second);
//                }
//            } else {
//                int a = random_int(nCandidates_ - 1);
//                int b = random_int(nCandidates_ - 1);
//                while (b == a)
//                    b = random_int(nCandidates_ - 1);
//                auto nw = measured_sketches_[a].genCrossover(measured_sketches_[b]);
//                if (nw.first) {
//                    ret.push_back(nw.second);
//                }
//            }
//        }
//    }
    return ret;
}

std::vector<Schedule>
AutoSchedule::genSchedules(const std::vector<Sketch> &sketches) {
    size_t n = sketches.size();
    std::vector<Schedule> ret(n);
    //#pragma omp parallel for
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

std::vector<std::vector<double>>
AutoSchedule::genFeatures(const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    std::vector<std::vector<double>> ret(n);
    //#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        try {
            ret[i] = fixedLengthFeature(schedules[i].ast());
        } catch (const std::exception &e) {
            // OpenMP threads won't report an exception message
            std::cerr << "ERROR: " << e.what() << std::endl;
            exit(-1);
        }
    }
    return ret;
}

std::vector<double>
AutoSchedule::testAndAdd(const std::vector<Sketch> &sketches,
                         const std::vector<Schedule> &schedules) {
    size_t n = schedules.size();
    ASSERT(sketches.size() == n);
    std::vector<double> times = measure(schedules);
    for (size_t i = 0; i < n; i++) {
        if (measured_sketches_.size() < nCandidates_) {
            measured_sketches_.emplace_back(sketches[i]);
            measured_sketches_.back().setTime(times[i]);
            std::push_heap(measured_sketches_.begin(),
                           measured_sketches_.end());
        } else if (times[i] < measured_sketches_[0].time()) {
            std::pop_heap(measured_sketches_.begin(), measured_sketches_.end());
            measured_sketches_.back() = sketches[i];
            measured_sketches_.back().setTime(times[i]);
            std::push_heap(measured_sketches_.begin(),
                           measured_sketches_.end());
        }
        mn_ = std::min(times[i], mn_);
    }
    std::cout << "min " << mn_ << " max " << measured_sketches_[0].time() << std::endl;
    return times;
}

Schedule AutoSchedule::getBestSchedule() {
    int best = 0;
    int time = measured_sketches_[0].time();
    for (size_t i = 0; i < measured_sketches_.size(); i++) {
        if (measured_sketches_[i].time() < time) {
            time = measured_sketches_[i].time();
            best = i;
        }
    }
    return measured_sketches_[best].genSchedule(original_);
}

std::vector<Sketch> AutoSchedule::GetInitPopulation(size_t n) {
    std::vector<Sketch> ret;
    std::set<size_t> used;
    std::vector<std::mt19937> gens;
    for (int i = 0; i < n; i++) {
        gens.push_back(std::mt19937(rand_gen()));
    }
    int iter = 0;
    while (ret.size() < n) {
        std::vector<Sketch> now(n);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            now[i] =
                baseSketches_[random_int(baseSketches_.size() - 1, gens[i])]
                    .genRandAnnotation(gens[i]);
        }
        for (int i = 0; i < n; i++) {
            size_t h = now[i].hash();
            if (!used.count(h)) {
                used.insert(h);
                ret.push_back(now[i]);
            }
        }
        if (++iter > 10) {
            break;
        }
    }
    size_t n_measured = std::min(size_t(n * 0.2), measured_sketches_.size());
    for (int i = 0; i < n_measured; i++) {
        size_t h = measured_sketches_[i].hash();
        if (!used.count(h)) {
            ret.push_back(measured_sketches_[i]);
        }
    }
    return ret;
}

std::vector<Sketch> AutoSchedule::EvolutionarySearch(std::vector<Sketch> init) {

    return std::vector<Sketch>();
}

} // namespace ir
