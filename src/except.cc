#include <iostream>
#include <mutex>
#include <unordered_map>

#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <config.h>
#include <debug.h>
#include <except.h>
#include <schedule.h>
#include <stmt.h>

namespace freetensor {

InvalidSchedule::InvalidSchedule(const Stmt &ast, const std::string &msg,
                                 std::source_location loc)
    : InvalidSchedule(FT_MSG << "Apply schedule on this AST is invalid: \n\n"
                             << toString(ast, Config::prettyPrint(), true)
                             << "\nThe reason is: " << msg,
                      loc) {}

InvalidSchedule::InvalidSchedule(const Ref<ScheduleLogItem> &log,
                                 const Stmt &ast, const std::string &msg,
                                 std::source_location loc)
    : InvalidSchedule(FT_MSG << "Apply schedule " << *log
                             << " on this AST is invalid: \n\n"
                             << toString(ast, Config::prettyPrint(), true)
                             << "\nThe reason is: " << msg,
                      loc) {}

InterruptExcept::InterruptExcept() : Error("Interrupted (Ctrl+C)") {
    // Raise SIGINT as normal. But if called from Python, nothing will happen,
    // because Python sets to ignore SIGINT
    kill(getpid(), SIGINT);
}

void reportWarning(const std::string &msg) {
    static std::unordered_map<std::string, int> reportCnt;
    static std::mutex lock;

    if (Config::werror()) {
        throw Error(msg);
    }

    std::lock_guard<std::mutex> guard(lock);
    int cnt = ++reportCnt[msg];
    if (cnt <= 2) {
        std::cerr << msg << std::endl;
    }
    if (cnt == 2) {
        std::cerr << "[INFO] Further identical warnings will be suppressed"
                  << std::endl;
    }
}

} // namespace freetensor
