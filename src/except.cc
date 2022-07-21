#include <mutex>
#include <unordered_map>

#include <config.h>
#include <debug.h>
#include <except.h>
#include <schedule.h>
#include <stmt.h>

namespace freetensor {

InvalidSchedule::InvalidSchedule(const Stmt &ast, const std::string &msg)
    : InvalidSchedule("Apply schedule on this AST is invalid: \n\n" +
                      toString(ast, Config::prettyPrint(), true) +
                      "\nThe reason is: " + msg) {}

InvalidSchedule::InvalidSchedule(const Ref<ScheduleLogItem> &log,
                                 const Stmt &ast, const std::string &msg)
    : InvalidSchedule("Apply schedule " + toString(*log) +
                      " on this AST is invalid: \n\n" +
                      toString(ast, Config::prettyPrint(), true) +
                      "\nThe reason is: " + msg) {}

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
        std::cerr << "[INFO] Further identical warnings will be supressed"
                  << std::endl;
    }
}

} // namespace freetensor
