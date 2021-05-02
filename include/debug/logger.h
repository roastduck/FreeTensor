#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>

namespace ir {

class Logger : public std::ostream {
    std::ostream &os;
    bool enable_ = true;

    static Logger instance_;

  public:
    Logger() : os(std::cerr) {}

    void enable() { enable_ = true; }
    void disable() { enable_ = false; }

    template <class T> Logger &operator<<(T &&x) {
        if (enable_) {
            os << x;
        }
        return *this;
    }

    Logger &operator<<(std::ostream &(*manip)(std::ostream &)) {
        if (enable_) {
            manip(os);
        }
        return *this;
    }

    static Logger &instance() { return instance_; }
};

inline Logger &logger() { return Logger::instance(); }

} // namespace ir

#endif // LOGGER_H
