#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unistd.h>

#include <config.h>
#include <except.h>
#include <ffi.h>
#include <opt.h>

namespace ir {

using namespace pybind11::literals;

static Opt<bool> getBoolEnv(const char *name) {
    static std::mutex lock;
    std::lock_guard<std::mutex> guard(lock); // getenv is not thread safe
    char *_env = getenv(name);
    if (_env == nullptr) {
        return nullptr;
    }
    std::string env(_env);
    if (env == "true" || env == "yes" || env == "on") {
        return Opt<bool>::make(true);
    } else if (env == "false" || env == "no" || env == "off") {
        return Opt<bool>::make(false);
    } else {
        ERROR((std::string) "Value of " + name +
              " must be true/yes/on or false/no/off");
    }
}

void init_ffi_config(py::module_ &m) {
    Config::setPrettyPrint(isatty(fileno(stdout)));

    if (auto flag = getBoolEnv("IR_PRETTY_PRINT"); flag.isValid()) {
        Config::setPrettyPrint(*flag);
    }
    if (auto flag = getBoolEnv("IR_PRINT_ALL_ID"); flag.isValid()) {
        Config::setPrintAllId(*flag);
    }

    m.def("with_mkl", Config::withMKL);
    m.def("set_pretty_print", Config::setPrettyPrint, "flag"_a = true);
    m.def("pretty_print", Config::prettyPrint);
    m.def("set_print_all_id", Config::setPrintAllId, "flag"_a = true);
    m.def("print_all_id", Config::printAllId);
}

} // namespace ir
