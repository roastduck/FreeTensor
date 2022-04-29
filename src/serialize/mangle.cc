#include <cctype>

#include <serialize/mangle.h>

namespace freetensor {

std::string mangle(const std::string &name) {
    std::string code;
    code.reserve(name.size() + 1);
    code += '_'; // Prepend an underscore to avoid conflicts with keywords in
                 // target languages
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
