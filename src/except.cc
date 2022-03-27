#include <config.h>
#include <debug.h>
#include <except.h>
#include <stmt.h>

namespace ir {

InvalidSchedule::InvalidSchedule(const std::string &msg, const Stmt &ast)
    : InvalidSchedule("Apply schedule on this AST is invalid: \n\n" +
                      toString(ast, Config::prettyPrint(), true) +
                      "\nThe reason is: " + msg) {}

} // namespace ir
