#include <mutex>
#include <unordered_map>

#include <config.h>
#include <debug.h>
#include <except.h>
#include <stmt.h>

namespace ir {

InvalidSchedule::InvalidSchedule(const std::string &msg, const Stmt &ast)
    : InvalidSchedule("Apply schedule on this AST is invalid: \n\n" +
                      toString(ast, Config::prettyPrint(), true) +
                      "\nThe reason is: " + msg) {}

void reportWarning(const std::string &msg) {
    static std::unordered_map<std::string, int> reportCnt;
    static std::mutex lock;

    std::lock_guard<std::mutex> guard(lock);
    int cnt = ++reportCnt[msg];
    if (cnt <= 2) {
        std::cerr << msg << std::endl;
    }
    if (cnt == 2) {
        std::cerr << "[NOTE] Further identical warnings will be supressed"
                  << std::endl;
    }
}

} // namespace ir
