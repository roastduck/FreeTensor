#ifndef EXCEPT_H
#define EXCEPT_H

#include <iostream>
#include <stdexcept>
#include <string>

namespace ir {

class Error : public std::runtime_error {
  public:
    Error(const std::string &msg) : std::runtime_error(msg) {}
};

class InvalidSchedule : public Error {
  public:
    InvalidSchedule(const std::string &msg) : Error(msg) {}
};

class DriverError : public Error {
  public:
    DriverError(const std::string &msg) : Error(msg) {}
};

class InvalidProgram : public Error {
  public:
    InvalidProgram(const std::string &msg) : Error(msg) {}
};

class AssertAlwaysFalse : public InvalidProgram {
  public:
    AssertAlwaysFalse(const std::string &msg) : InvalidProgram(msg) {}
};

#define ERROR(msg)                                                             \
    {                                                                          \
        throw ::ir::Error((std::string) "[ERROR] " __FILE__ ":" +              \
                          std::to_string(__LINE__) + ": " + (msg));            \
    }

#define WARNING(msg)                                                           \
    {                                                                          \
        std::cerr << ((std::string) "[WARING] " __FILE__ ":" +                 \
                      std::to_string(__LINE__) + ": " + (msg))                 \
                  << std::endl;                                                \
    }

#define ASSERT(expr)                                                           \
    {                                                                          \
        if (!(expr)) {                                                         \
            ERROR("Assertion false: " #expr)                                   \
        }                                                                      \
    }

} // namespace ir

#endif // EXCEPT_H
