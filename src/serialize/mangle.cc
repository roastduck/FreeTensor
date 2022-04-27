#include <cctype>

#include <serialize/mangle.h>

namespace freetensor {

std::string mangle(const std::string &name) {
    std::string code;
    code.reserve(name.size());
    for (char c : name) {
        if (isalnum(c)) {
            code += c;
        } else if (c == '_') {
            code += "__";
        } else {
            code += "_" + std::to_string((int)c);
        }
    }
    return code;
}

} // namespace freetensor
