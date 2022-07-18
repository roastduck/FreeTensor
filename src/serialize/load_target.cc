#include <serialize/load_target.h>

#include <string>
#include <sstream>


namespace freetensor{

Ref<Target> loadTarget(const std::string &txt) {

    std::istringstream iss(txt);

    auto ret = Ref<GPU>::make();
    std::string type, has_extra;
    bool arg0;
    int arg1, arg2;

    ASSERT(iss >> type >> arg0);
    ret->setUseNativeArch(arg0);

    ASSERT(type.length() > 0);

    switch (type[0]) {
    case 'G':
        ASSERT(iss >> has_extra);
        if (has_extra == ":") {
            ASSERT(iss >> arg1 >> arg2);
            ret->setComputeCapability(arg1, arg2);
        }
        break; // GPU
    case 'C':
        break; // CPU

    default:
        ASSERT(false);
    }

    return ret.as<Target>();
}

} // namespace freetensor

