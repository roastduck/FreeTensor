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

class StmtNode;
template <class T> class Ref;
typedef Ref<StmtNode> Stmt;

class InvalidSchedule : public Error {
  public:
    InvalidSchedule(const std::string &msg) : Error(msg) {}
    InvalidSchedule(const std::string &msg, const Stmt &ast);
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

void reportWarning(const std::string &msg);

#define ERROR(msg)                                                             \
    {                                                                          \
        throw ::ir::Error((std::string) "[ERROR] " __FILE__ ":" +              \
                          std::to_string(__LINE__) + ": " + (msg));            \
    }

#define WARNING(msg)                                                           \
    {                                                                          \
        reportWarning((std::string) "[WARNING] " __FILE__ ":" +                \
                      std::to_string(__LINE__) + ": " + (msg));                \
    }

#define ASSERT(expr)                                                           \
    {                                                                          \
        if (!(expr)) {                                                         \
            ERROR("Assertion false: " #expr)                                   \
        }                                                                      \
    }

} // namespace ir

#endif // EXCEPT_H
