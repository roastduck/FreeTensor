#ifndef FREE_TENSOR_LOGGER_H
#define FREE_TENSOR_LOGGER_H

#include <iostream>
#include <mutex>
#include <thread>

namespace freetensor {

class LogCtrl {
    bool enable_ = true;

    static LogCtrl instance_;

  public:
    bool isEnabled() const { return enable_; }
    void enable() { enable_ = true; }
    void disable() { enable_ = false; }

    static LogCtrl &instance() { return instance_; }
};

class Logger {
    std::ostream &os_;

    static std::mutex lock_;

  public:
    Logger() : os_(std::cerr) {
        lock_.lock();
        if (LogCtrl::instance().isEnabled()) {
            os_ << "[tid " << std::hex << std::this_thread::get_id() << "] ";
        }
    }
    ~Logger() { lock_.unlock(); }

    Logger(const Logger &) = delete;
    Logger(Logger &&other) = default;
    Logger &operator=(const Logger &) = delete;

    void enable() { LogCtrl::instance().enable(); }
    void disable() { LogCtrl::instance().disable(); }

    template <class T> Logger &operator<<(T &&x) {
        if (LogCtrl::instance().isEnabled()) {
            os_ << x;
        }
        return *this;
    }

    Logger &operator<<(std::ostream &(*manip)(std::ostream &)) {
        if (LogCtrl::instance().isEnabled()) {
            manip(os_);
        }
        return *this;
    }
};

inline Logger logger() { return Logger(); }

} // namespace freetensor

#endif // FREE_TENSOR_LOGGER_H
