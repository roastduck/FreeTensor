#include <config.h>

#define NAME_(macro) #macro
#define NAME(macro) NAME_(macro)

namespace ir {

bool Config::prettyPrint_ = false;

std::string Config::withMKL() {
#ifdef WITH_MKL
    return NAME(WITH_MKL);
#else
    return "";
#endif
}

} // namespace ir
