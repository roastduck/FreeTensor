#ifndef FREE_TENSOR_SCHEDULE_LOG_H
#define FREE_TENSOR_SCHEDULE_LOG_H

#include <array>
#include <exception>
#include <iostream>
#include <variant>

#include <ast.h>
#include <serialize/to_string.h>
#include <shared_linked_list.h>

namespace freetensor {

enum class ScheduleType : int {
    Split = 0,
    Reorder,
    Merge,
    Fission,
    Fuse,
    Swap,
    Blend,
    Cache,
    CacheReduction,
    SetMemType,
    VarSplit,
    VarMerge,
    VarReorder,
    Inline,
    Parallelize,
    Unroll,
    Vectorize,
    SeparateTail,
    AsMatMul,
    Permute,
    // ------
    NumTypes,
};

constexpr std::array scheduleTypeNames = {
    "split",        "reorder",   "merge",
    "fission",      "fuse",      "swap",
    "blend",        "cache",     "cache_reduction",
    "set_mem_type", "var_split", "var_merge",
    "var_reorder",  "inline",    "parallelize",
    "unroll",       "vectorize", "separate_tail",
    "as_matmul",    "permute",
};
static_assert(scheduleTypeNames.size() == (size_t)ScheduleType::NumTypes);

inline std::ostream &operator<<(std::ostream &os, ScheduleType type) {
    return os << scheduleTypeNames.at((size_t)type);
}

/**
 * Log of a schedule
 *
 * An object of this class records what schedule a user applies and what
 * parameters are used, which can be hashed and compared with other
 * `ScheduleLogItem`s.
 *
 * Inherit this class for specific parameters and result types. Schedule result
 * can also be added in subclasses, but not compared
 */
class ScheduleLogItem {
  public:
    virtual ~ScheduleLogItem() {}
    virtual ScheduleType type() const = 0;
    virtual std::string toString() const = 0;
    virtual size_t hash() const = 0;
    virtual bool equals(const ScheduleLogItem &other) const = 0;
    virtual void run() = 0;
};

/**
 * Template of a specialized `ScheduleLogItem` of a particular type of schedule
 */
template <ScheduleType TYPE, class _Invocable, class _Params, class _Result>
class ScheduleLogItemImpl : public ScheduleLogItem {
  protected:
    // Types defined for subclasses
    typedef _Invocable Invocable;
    typedef _Params Params;
    typedef _Result Result;

    Invocable doSchedule_;
    Params params_;
    std::variant<std::nullopt_t, Result, std::exception_ptr> result_ =
        std::nullopt;

  public:
    ScheduleLogItemImpl(const Invocable &doSchedule, const Params &params)
        : doSchedule_(doSchedule), params_(params) {}

    ScheduleType type() const override { return TYPE; }

    std::string toString() const override {
        std::ostringstream os;
        os << std::boolalpha << type() << '(' << params_ << ')';
        return os.str();
    }

    size_t hash() const override { return std::hash<Params>()(params_); }

    bool equals(const ScheduleLogItem &other) const override {
        if (other.type() != type()) {
            return false;
        }
        return ((const ScheduleLogItemImpl &)other).params_ == params_;
    }

    /**
     * Run a schedule and save its result or its exception
     */
    void run() override {
        if (std::holds_alternative<std::nullopt_t>(result_)) {
            try {
                result_ = std::apply(doSchedule_, params_);
            } catch (...) {
                result_ = std::current_exception();
            }
        }
    }

    /**
     * Get a saved result or re-throw an exception
     */
    Result getResult() const {
        if (std::holds_alternative<std::nullopt_t>(result_)) {
            ERROR("BUG: The schedule log is not run yet");
        } else if (std::holds_alternative<Result>(result_)) {
            return std::get<Result>(result_);
        } else {
            ASSERT(std::holds_alternative<std::exception_ptr>(result_));
            std::rethrow_exception(std::get<std::exception_ptr>(result_));
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const ScheduleLogItem &log) {
    return os << log.toString();
}

struct ScheduleLogItemEqual {
    bool operator()(const Ref<ScheduleLogItem> &lhs,
                    const Ref<ScheduleLogItem> &rhs) const {
        return lhs->equals(*rhs);
    }
};

struct ScheduleLogItemHash {
    bool operator()(const Ref<ScheduleLogItem> &item) const {
        return item->hash();
    }
};

typedef SharedLinkedList<Ref<ScheduleLogItem>, ScheduleLogItemHash,
                         ScheduleLogItemEqual>
    ScheduleLog;

} // namespace freetensor

#endif // FREE_TENSOR_SCHEDULE_LOG_H
