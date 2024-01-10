#ifndef FREE_TENSOR_EXCEPT_H
#define FREE_TENSOR_EXCEPT_H

#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>

namespace freetensor {

class MessageBuilder {
    std::ostringstream os_;

  public:
    template <typename T> MessageBuilder &operator<<(const T &obj) {
        os_ << obj;
        return *this;
    }

    operator std::string() const { return os_.str(); }
};

#define FT_MSG MessageBuilder()

class Error : public std::runtime_error {
  public:
    // NOTE: `source_location` is intended to have a small size and can be
    // copied efficiently.
    Error(const std::string &msg,
          std::source_location loc = std::source_location::current())
        : std::runtime_error(FT_MSG << loc.file_name() << ":" << loc.line()
                                    << ": " << msg) {}
};

class StmtNode;
class ScheduleLogItem;
template <class T> class Ref;
typedef Ref<StmtNode> Stmt;

class InvalidSchedule : public Error {
  public:
    InvalidSchedule(const std::string &msg,
                    std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
    InvalidSchedule(const Stmt &ast, const std::string &msg,
                    std::source_location loc = std::source_location::current());
    InvalidSchedule(const Ref<ScheduleLogItem> &log, const Stmt &ast,
                    const std::string &msg,
                    std::source_location loc = std::source_location::current());
};

class InvalidAutoGrad : public Error {
  public:
    InvalidAutoGrad(const std::string &msg,
                    std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

/**
 * Invalid configurations to Driver, or error reported by backend compilers or
 * the OS
 */
class DriverError : public Error {
  public:
    DriverError(const std::string &msg,
                std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

/**
 * Unable to pass input data or receive output data from the compiled program
 */
class InvalidIO : public Error {
  public:
    InvalidIO(const std::string &msg,
              std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

/**
 * The program is ill-formed
 */
class InvalidProgram : public Error {
  public:
    InvalidProgram(const std::string &msg,
                   std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

class SymbolNotFound : public Error {
  public:
    SymbolNotFound(const std::string &msg,
                   std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

class AssertAlwaysFalse : public InvalidProgram {
  public:
    AssertAlwaysFalse(
        const std::string &msg,
        std::source_location loc = std::source_location::current())
        : InvalidProgram(msg, loc) {}
};

class ParserError : public Error {
  public:
    ParserError(const std::string &msg,
                std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
};

class UnexpectedQueryResult : public Error {
  public:
    UnexpectedQueryResult(
        const std::string &msg,
        std::source_location loc = std::source_location::current())
        : Error(msg, loc) {}
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
        throw ::freetensor::Error(msg);                                        \
    } while (0)

#define WARNING(msg)                                                           \
    do {                                                                       \
        reportWarning(FT_MSG << "[WARNING] " __FILE__ ":" << __LINE__ << ": "  \
                             << std::string(msg));                             \
    } while (0)

#define ASSERT(expr)                                                           \
    do {                                                                       \
        if (!(expr))                                                           \
            ERROR("Assertion false: " #expr);                                  \
    } while (0)

} // namespace freetensor

#endif // FREE_TENSOR_EXCEPT_H
