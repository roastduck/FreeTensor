#ifndef FREE_TENSOR_EXCEPT_H
#define FREE_TENSOR_EXCEPT_H

#include <iostream>
#include <stdexcept>
#include <string>

namespace freetensor {

class Error : public std::runtime_error {
  public:
    Error(const std::string &msg) : std::runtime_error(msg) {}
};

class StmtNode;
class ScheduleLogItem;
template <class T> class Ref;
typedef Ref<StmtNode> Stmt;

class InvalidSchedule : public Error {
  public:
    InvalidSchedule(const std::string &msg) : Error(msg) {}
    InvalidSchedule(const Stmt &ast, const std::string &msg);
    InvalidSchedule(const Ref<ScheduleLogItem> &log, const Stmt &ast,
                    const std::string &msg);
};

class InvalidAutoGrad : public Error {
  public:
    InvalidAutoGrad(const std::string &msg) : Error(msg) {}
};

/**
 * Invalid configurations to Driver, or error reported by backend compilers or
 * the OS
 */
class DriverError : public Error {
  public:
    DriverError(const std::string &msg) : Error(msg) {}
};

/**
 * Unable to pass input data or receive output data from the compiled program
 */
class InvalidIO : public Error {
  public:
    InvalidIO(const std::string &msg) : Error(msg) {}
};

/**
 * The program is ill-formed
 */
class InvalidProgram : public Error {
  public:
    InvalidProgram(const std::string &msg) : Error(msg) {}
};

class SymbolNotFound : public Error {
  public:
    SymbolNotFound(const std::string &msg) : Error(msg) {}
};

class AssertAlwaysFalse : public InvalidProgram {
  public:
    AssertAlwaysFalse(const std::string &msg) : InvalidProgram(msg) {}
};

class ParserError : public Error {
  public:
    ParserError(const std::string &msg) : Error(msg) {}
};

class UnexpectedQueryResult : public Error {
  public:
    UnexpectedQueryResult(const std::string &msg) : Error(msg) {}
};

/**
 * SIGINT as an exception
 *
 * This is required because Python has its own signal handler. See
 * https://docs.python.org/3/c-api/exceptions.html#signal-handling
 *
 * If not invoking from Python, throwing this exception is equivalent to raise
 * SIGINT
 *
 * A special adaptor in `ffi/except.cc` is implemented for this exception.
 * See
 * https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions
 * for details
 */
class InterruptExcept : public Error {
  public:
    InterruptExcept();
};

void reportWarning(const std::string &msg);

#define ERROR(msg)                                                             \
    do {                                                                       \
        throw ::freetensor::Error((std::string) "[ERROR] " __FILE__ ":" +      \
                                  std::to_string(__LINE__) + ": " + (msg));    \
    } while (0)

#define WARNING(msg)                                                           \
    do {                                                                       \
        reportWarning((std::string) "[WARNING] " __FILE__ ":" +                \
                      std::to_string(__LINE__) + ": " + (msg));                \
    } while (0)

#define ASSERT(expr)                                                           \
    do {                                                                       \
        if (!(expr))                                                           \
            ERROR("Assertion false: " #expr);                                  \
    } while (0)

} // namespace freetensor

#endif // FREE_TENSOR_EXCEPT_H
