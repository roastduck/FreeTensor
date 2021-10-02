#include <analyze/find_thread_bind.h>
#include <auto_schedule/rules/thread_bind.h>
#include <schedule.h>

namespace ir {
int ThreadBindRule::analyze(Schedule &schedule) {
    target = findThreadBind(schedule.ast());
    if (target.empty())
        return false;
    return 1;
}

SketchPart ThreadBindRule::genPart(int p) {
    return SketchPart(new ThreadBindPart(target, target_));
}

void ThreadBindPart::genRandAnnotation() {}

ThreadBindPart::ThreadBindPart(
    std::vector<std::pair<std::string, int>> targetFor, Ref<Target> target)
    : targetFor_(targetFor), target_(target) {}

void ThreadBindPart::apply(Schedule &schedule) {
    if (target_->type() == TargetType::GPU) {
        applyOnGPU(schedule);
    } else {
        applyOnCPU(schedule);
    }
}

void ThreadBindPart::applyOnCPU(Schedule &schedule) {
    for (size_t i = 0; i < targetFor_.size(); i++) {
        try {
            schedule.parallelize(targetFor_[i].first, "openmp");
            return;
        } catch (const InvalidSchedule &e) {
            // do nothing
        }
    }
}

// TODO: The result of applyOnGPU rely on the specific parameters
//       in the AST.
// TODO: Specific number should not be transferred.
void ThreadBindPart::applyOnGPU(Schedule &schedule) {
    int fuseNum = 0;
    int targetForLength = targetFor_.size();
    if (!targetForLength) {
        return;
    }
    std::string fusedFor = targetFor_[0].first;
    int fusedLength = targetFor_[0].second;
    try {
        while (fuseNum < targetForLength - 1) {
            fusedFor = schedule.fuse(fusedFor, targetFor_[fuseNum + 1].first);
            fusedLength *= targetFor_[fuseNum + 1].second;
            fuseNum++;
        }
    } catch (const InvalidSchedule &e) {
        // do nothing
    }
    if (fusedLength > BLOCK_MIN_NUM &&
        fusedLength <= BLOCK_MIN_NUM * THREAD_MAX_NUM) {
        try {
            std::pair<std::string, std::string> splittedFors =
                schedule.split(fusedFor, -1, BLOCK_MIN_NUM);
            schedule.parallelize(splittedFors.first, "blockIdx.x");
            schedule.parallelize(splittedFors.second, "threadIdx.x");
        } catch (const InvalidSchedule &e) {
            std::cerr << "Auto thread binding failed. 1" << '\n';
        }
    } else if (fusedLength > BLOCK_MIN_NUM * THREAD_MAX_NUM) {
        try {
            std::pair<std::string, std::string> splittedFors =
                schedule.split(fusedFor, THREAD_MAX_NUM);
            schedule.parallelize(splittedFors.first, "blockIdx.x");
            schedule.parallelize(splittedFors.second, "threadIdx.x");
        } catch (const InvalidSchedule &e) {
            std::cerr << "Auto thread binding failed. 2" << '\n';
        }
    } else if (fusedLength <= BLOCK_MIN_NUM) {
        try {
            schedule.parallelize(fusedFor, "blockIdx.x");
        } catch (const InvalidSchedule &e) {
            std::cerr << "Auto thread binding failed. 3" << '\n';
        }
    }
    // int cnt = 0;
    // size_t i = 0;
    // std::vector<std::string> parallel{"blockIdx.x",  "blockIdx.y",
    //                                   "blockIdx.z",  "threadIdx.x",
    //                                   "threadIdx.y", "threadIdx.z"};
    // int product = 1;
    // for (; i < targetFor_.size() && cnt < 3; i++) {
    //     try {
    //         schedule.parallelize(targetFor_[i].first, parallel[cnt]);
    //         cnt++;
    //     } catch (const InvalidSchedule &e) {
    //         // do nothing
    //     }
    // }
    // for (; i < targetFor_.size() && cnt < 6; i++) {
    //     try {
    //         if (product * targetFor_[i].second > THREAD_MAX_NUM) {
    //             std::pair<std::string, std::string> newLoops =
    //             schedule.split(
    //                 targetFor_[i].first, -1, THREAD_MAX_NUM / product);
    //             schedule.parallelize(newLoops.first, parallel[cnt]);
    //             return;
    //         } else {
    //             schedule.parallelize(targetFor_[i].first, parallel[cnt]);
    //             cnt++;
    //             product *= targetFor_[i].second;
    //             if (product > THREAD_MAX_NUM / 2) {
    //                 return;
    //             }
    //         }
    //     } catch (const std::exception &e) {
    //         // do nothing
    //     }
    // }
}

SketchPart ThreadBindPart::mutate() { return Ref<ThreadBindPart>::make(*this); }

SketchPart ThreadBindPart::crossover(const SketchPart &part) {
    return Ref<ThreadBindPart>::make(*this);
}

std::vector<int> ThreadBindPart::getAnnotation() const {
    return std::vector<int>(1, 1);
}

} // namespace ir