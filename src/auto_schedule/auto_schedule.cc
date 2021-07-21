#include <auto_schedule/auto_schedule.h>
#include <auto_schedule/rules/multi_level_tiling.h>
#include <auto_schedule/utils.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <driver.h>
#include <lower.h>

namespace ir {

void AutoSchedule::set_params(
    const std::vector<Array *> &args,
    const std::unordered_map<std::string, Array *> &kws) {
    args_ = args;
    kws_ = kws;
    params_set = true;
}

std::vector<double> AutoSchedule::measure(const std::vector<Sketch> &sketches) {
    // Compile in parallel, and measure sequentially
    // TODO: Parallel among computing nodes

    size_t n = sketches.size();
    std::vector<Ref<Driver>> drivers(n);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        auto func = lower(sketches[i].gen_schedule().func(), target);
        std::string code;
        if (target->type() == TargetType::GPU)
            code = codeGenCUDA(func);
        else
            code = codeGenCPU(func);
        drivers[i] = Ref<Driver>::make(Driver(func, code, device));
    }

    std::vector<double> times;
    times.reserve(n);
    for (size_t i = 0; i < n; i++) {
        drivers[i]->setParams(args_, kws_);
        times.emplace_back(drivers[i]->time(5, 20));
    }
    return times;
}

std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::init(int _n_candidates) {
    n_candidates = _n_candidates;
    if (!params_set) {
        ERROR("Please set params first");
    }

    Sketch sketch(schedule_);
    MultiLevelTilingRule rule;
    int n = rule.analyze(schedule_);
    std::cout << "Found" << n << std::endl;
    sketch.add_part(rule.gen_part(0));

    candidates.reserve(n_candidates);
    for (size_t i = 0; i < n_candidates; i++) {
        candidates.emplace_back(sketch.gen_rand_annotation());
    }
    std::vector<double> times = measure(candidates);
    for (size_t i = 0; i < n_candidates; i++) {
        candidates[i].time = times[i];
    }

    std::vector<std::vector<int>> annotations;
    annotations.reserve(candidates.size());
    for (auto &&candidate : candidates) {
        annotations.emplace_back(candidate.get_annotation());
    }

    std::sort(candidates.begin(), candidates.end());
    std::cout << "Initial: min " << candidates[0].time << " , max "
              << candidates[n_candidates - 1].time << std::endl;
    mn_ = candidates[0].time;
    std::make_heap(candidates.begin(), candidates.begin() + n_candidates);
    return std::make_pair(annotations, times);
}

std::vector<Sketch> AutoSchedule::get_random_sketches(size_t n) {
    std::vector<Sketch> ret;
    while (ret.size() < n) {
        int mut = random_int(1);
        if (mut) {
            auto nw = candidates[random_int(n_candidates - 1)].gen_mutation();
            if (nw.first) {
                ret.push_back(nw.second);
            }
        } else {
            int a = random_int(n_candidates - 1);
            int b = random_int(n_candidates - 1);
            while (b == a)
                b = random_int(n_candidates - 1);
            auto nw = candidates[a].gen_crossover(candidates[b]);
            if (nw.first) {
                ret.push_back(nw.second);
            }
        }
    }
    return ret;
}

std::pair<std::vector<std::vector<int>>, std::vector<double>>
AutoSchedule::test_and_add(const std::vector<Sketch> &sketches) {
    std::vector<double> times = measure(sketches);
    std::vector<std::vector<int>> annotations;
    annotations.reserve(sketches.size());
    for (size_t i = 0, iEnd = sketches.size(); i < iEnd; i++) {
        annotations.emplace_back(sketches[i].get_annotation());
        if (times[i] < candidates[0].time) {
            std::pop_heap(candidates.begin(),
                          candidates.begin() + n_candidates);
            candidates[n_candidates - 1] = sketches[i];
            candidates[n_candidates - 1].time = times[i];
            std::push_heap(candidates.begin(),
                           candidates.begin() + n_candidates);
            mn_ = std::min(times[i], mn_);
        }
    }
    std::cout << "min " << mn_ << " max " << candidates[0].time << std::endl;
    return std::make_pair(annotations, times);
}

Schedule AutoSchedule::get_best_schedule() {
    int best = 0;
    int time = candidates[0].time;
    for (size_t i = 0; i < candidates.size(); i++) {
        if (candidates[i].time < time) {
            time = candidates[i].time;
            best = i;
        }
    }
    return candidates[best].gen_schedule();
}

} // namespace ir
